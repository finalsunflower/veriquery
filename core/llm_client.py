"""
LLM client — unified interface for HuggingFace local model inference.

Provides lazy-loaded model management, RAG answer generation with prompt
engineering, answer post-processing, and a thread-safe singleton factory.

Usage:
    from core.llm_client import get_llm_client
    client = get_llm_client(settings)
    answer = client.generate_answer(query="NE5532供电电压?", context="...")
"""

import logging
import platform
import re
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class LLMClient:
    """HuggingFace LLM client with lazy loading, quantization, and RAG generation.

    Model lifecycle:
        __init__() saves settings only; model loads on first _generate() call.
        _load_model() checks VRAM, selects quantization strategy, loads model.
    """

    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

    def _load_model(self):
        """Lazy-load the HuggingFace LLM model with VRAM check and quantization.

        Quantization strategy:
          - Windows + quantize → float16 (bitsandbytes unavailable on Windows)
          - Linux + CUDA + 4bit → BitsAndBytesConfig NF4
          - Linux + CUDA + 8bit → BitsAndBytesConfig INT8
          - CUDA + no quantize → float16 + device_map="auto"
          - CPU → float32

        Raises:
            RuntimeError: If model loading fails.
        """
        if self._model_loaded:
            return

        try:
            from core.memory_manager import get_memory_manager
            memory_manager = get_memory_manager(self.settings)

            memory_manager.log_memory_usage("LLM加载前")
            memory_manager.check_and_cleanup(
                required_memory_gb=1.0,
                cleanup_threshold_gb=1.5,
            )

            model_path = self.settings.LLM_MODEL
            device = self.settings.LLM_DEVICE
            use_quantization = self.settings.LLM_QUANTIZE
            quantization_bits = self.settings.LLM_QUANTIZATION_BITS

            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"

            load_kwargs = {
                "local_files_only": True,
                "trust_remote_code": True,
            }

            quantization_config = None

            if use_quantization and device == "cuda":
                is_windows = platform.system() == "Windows"

                if is_windows:
                    logger.info("Windows detected, using float16 (bitsandbytes unavailable)")
                    load_kwargs["torch_dtype"] = torch.float16
                    load_kwargs["device_map"] = "auto"
                else:
                    from transformers import BitsAndBytesConfig

                    if quantization_bits == 4:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            llm_int8_enable_fp32_cpu_offload=True,
                        )
                        logger.info("Using 4-bit NF4 quantization with CPU offload")
                    elif quantization_bits == 8:
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_enable_fp32_cpu_offload=True,
                        )
                        logger.info("Using 8-bit quantization with CPU offload")

                    if quantization_config:
                        load_kwargs["quantization_config"] = quantization_config
                        load_kwargs["device_map"] = "auto"

            if device == "cuda":
                if "device_map" not in load_kwargs:
                    load_kwargs["device_map"] = "auto"
                if "torch_dtype" not in load_kwargs:
                    load_kwargs["torch_dtype"] = torch.float16
            else:
                load_kwargs["torch_dtype"] = torch.float32

            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._model_loaded = True

            memory_manager.log_memory_usage("LLM加载后")
            logger.info(f"LLM model loaded: {model_path}")

        except Exception as e:
            logger.error(f"LLM model loading failed: {e}")
            self.model = None
            self.tokenizer = None
            self._model_loaded = False
            raise RuntimeError(f"Failed to create LLM client: {e}") from e

    def _generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Run model inference: tokenize → generate → decode → truncate noise.

        Args:
            prompt: Input text for the model.
            max_new_tokens: Max tokens to generate (hard cap at 256).
            temperature: Sampling temperature (hard cap at 0.7).

        Returns:
            Generated text with conversation-format noise truncated.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        self._load_model()

        if not self.model or not self.tokenizer:
            raise RuntimeError("LLM model not loaded, check system configuration")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": min(max_new_tokens, 256),
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.15,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
        }

        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = min(temperature, 0.7)
            gen_kwargs["top_p"] = 0.9
            gen_kwargs["top_k"] = 50
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        for stop_word in ["assistant", "user:", "system:", "\n\n\n", "###", "【答案】"]:
            if stop_word.lower() in answer.lower():
                idx = answer.lower().find(stop_word.lower())
                if idx > 5:
                    answer = answer[:idx]
                    break

        return answer.strip()

    def generate_answer(
        self,
        query: str,
        context: str,
        max_new_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """Generate an answer for a RAG query using context and prompt engineering.

        Pipeline: _build_prompt() → _generate() → _clean_answer()

        Args:
            query: User question.
            context: Retrieved context text.
            max_new_tokens: Override for max generation tokens (defaults to settings).
            temperature: Override for sampling temperature (defaults to settings).

        Returns:
            Cleaned answer text.
        """
        if max_new_tokens is None:
            max_new_tokens = self.settings.LLM_MAX_TOKENS
        if temperature is None:
            temperature = self.settings.LLM_TEMPERATURE

        if max_new_tokens <= 0 or max_new_tokens > 2048:
            max_new_tokens = self.settings.LLM_MAX_TOKENS
        if temperature < 0 or temperature > 2.0:
            temperature = self.settings.LLM_TEMPERATURE

        try:
            prompt = self._build_prompt(query, context)
            answer = self._generate(prompt, max_new_tokens, temperature)
            answer = self._clean_answer(answer)
            if not answer or len(answer.strip()) < 5:
                logger.warning("LLM首次生成答案为空或过短，降低温度重试")
                answer = self._generate(prompt, max_new_tokens, max(0.1, temperature - 0.2))
                answer = self._clean_answer(answer)
            return answer
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            raise

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the RAG prompt with context, query, and chip-name constraints."""
        chip_pattern = re.compile(
            r'('
            r'(?:NE|LM|SN|CD|STM32|AD|MAX|TL|OP|UA|MC|ATmega|ATtiny)[A-Z0-9.-]+'
            r'|\d{2}[A-Z]{2,}\d+[A-Z0-9-]*'
            r')',
            re.IGNORECASE,
        )
        chip_match = chip_pattern.search(query)
        chip_name = chip_match.group(1).upper() if chip_match else None

        chip_constraint = ""
        if chip_name:
            chip_constraint = f"""
【核心约束 - 必须严格遵守】
1. 用户询问的芯片型号是 {chip_name}，答案中必须使用这个确切的型号名称
2. 禁止修改、变形或错误拼写芯片型号（如将{chip_name}写成其他形式）
3. 答案开头必须明确指出芯片型号："{chip_name}的..."
4. 只回答{chip_name}的参数，忽略技术资料中其他芯片的信息
5. 数值和单位必须直接从技术资料中提取，禁止编造
"""

        return f"""你是电子元器件技术文档分析专家。请严格根据以下技术资料回答问题。

【技术资料】
{context}

【用户问题】
{query}
{chip_constraint}
【回答格式要求】
1. 直接给出答案，以芯片型号开头
2. 回答控制在2-3句话以内
3. 只使用技术资料中的数据，禁止编造
4. 数值必须带正确单位（V=电压，A=电流，℃=温度）
5. 不要输出分析过程，只输出最终答案

【答案】"""

    def _clean_answer(self, answer: str) -> str:
        """Clean LLM output: remove format noise, boilerplate, and duplicate sentences.

        Steps:
            1. Remove <think...</think > blocks
            2. Remove thinking/analysis prefixes and their entire block
            3. Truncate conversation markers (assistant/user/system)
            4. Strip Markdown/HTML formatting
            5. Remove boilerplate prefixes
            6. Decode HTML entities
            7. Deduplicate sentences via 20-char prefix hash
            8. Fallback to safe cleaning if aggressive cleaning removes too much content
        """
        raw_answer = answer

        answer = re.sub(r'<think\s*>.*?</think\s*>', '', answer, flags=re.DOTALL | re.IGNORECASE)

        thinking_prefixes = [
            r'\d*\s*Thinking\s*Process\s*:\s*',
            r'\d*\s*Analyze\s+the\s+Request\s*:\s*',
            r'\d*\s*Let(?:\'s|\s+me)\s+(?:think|analyze|reason|break\s+down)\s*[:：]?\s*',
            r'\d*\s*Step\s*\d+\s*[:：]\s*Analysis\s*',
        ]
        for prefix in thinking_prefixes:
            m = re.match(prefix, answer, re.IGNORECASE)
            if m:
                after_prefix = answer[m.end():]
                actual_answer_match = re.search(
                    r'(?:^|\n)\s*('
                    r'(?:NE|LM|SN|CD|STM32?|AD|MAX|TL|OP|UA|MC|AT)\w*(?:的|是|工作|电源|电压|供电|输入|输出|最大|最小|范围|绝对)'
                    r'|根据.*?(?:规格|数据|资料|文档|手册)'
                    r'|[±]?\d+\.?\d*\s*[VvAa][\s，,。.；;]'
                    r'|该(?:芯片|器件|元件)'
                    r'|Supply\s+voltage'
                    r')',
                    after_prefix, re.IGNORECASE,
                )
                if actual_answer_match:
                    start_pos = actual_answer_match.start()
                    if after_prefix[start_pos] == '\n':
                        start_pos += 1
                    answer = after_prefix[start_pos:]
                else:
                    answer = ""
                break

        answer = re.sub(
            r'^\d*\s*\*\s*(?:Step|Role|Task|Question|Analysis|Reasoning|User\s+Question)\s*\d*\s*[:：].*$',
            '', answer, flags=re.IGNORECASE | re.MULTILINE,
        )

        for stop_word in ["\n\n\n"]:
            if stop_word in answer:
                idx = answer.find(stop_word)
                answer = answer[:idx]

        for stop_word in ["assistant:", "user:", "system:"]:
            lower = answer.lower()
            idx = lower.find(stop_word)
            if idx > 20:
                answer = answer[:idx]

        idx_assistant = answer.lower().find("assistant")
        if idx_assistant > 0 and idx_assistant < len(answer) - 12:
            after = answer[idx_assistant:idx_assistant + 12].lower()
            if after in ("assistant", "assistant\n", "assistant "):
                answer = answer[:idx_assistant]

        for marker in ["###", "【答案】"]:
            lower = answer.lower()
            idx = lower.find(marker.lower())
            if idx > 30:
                answer = answer[:idx]

        answer = re.sub(r'^(?:Wait|Oh\s+wait|Let\s+me\s+think|I\s+need\s+to)[,.]?\s+', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'^\d+\s+(?=[A-Za-z\u4e00-\u9fff])', '', answer)

        answer = answer.lstrip('\n*-. ')

        answer = re.sub(r"\*\*", "", answer)
        answer = re.sub(r"<[^>]+>", "", answer)
        answer = re.sub(r"```markdown\s*", "", answer)
        answer = re.sub(r"```\s*", "", answer)
        answer = re.sub(r'^\s*(?:Solution|Answer|Result)\s*[:：]?\s*$', '', answer, flags=re.IGNORECASE | re.MULTILINE)

        answer = re.sub(r"根据.*?分析如下：", "", answer, flags=re.IGNORECASE | re.DOTALL)
        answer = re.sub(r"作为[^。！？]+专家[^。！？]*[，,。！？]", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"我需基于[^。！？]+[。！？]", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"以下是针对[^。！？]+[：:]", "", answer, flags=re.IGNORECASE)

        answer = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', answer)
        answer = re.sub(r'(\d+)\.\n(\d+)', r'\1.\2', answer)
        answer = re.sub(r'(\d+)\s+(V|A|℃|°C|mA|mV|μA|kHz|MHz)', r'\1\2', answer, flags=re.IGNORECASE)
        answer = re.sub(r'(\d+)\.\s*(V|A|℃|°C|mA|mV|μA|kHz|MHz)', r'\1\2', answer, flags=re.IGNORECASE)

        html_entities = {
            "&lt;": "<",
            "&gt;": ">",
            "&amp;": "&",
            "&quot;": '"',
            "&apos;": "'",
        }
        for entity, char in html_entities.items():
            answer = answer.replace(entity, char)

        sentences = re.split(r'(?<=[。！？.!?])|\n', answer)
        seen = set()
        unique_sentences = []

        for sent in sentences:
            sent_stripped = sent.strip()
            if sent_stripped and len(sent_stripped) > 6:
                sent_normalized = re.sub(r'\s+', '', sent_stripped.lower())
                if len(sent_normalized) > 4:
                    similarity_hash = sent_normalized[:20] if len(sent_normalized) > 20 else sent_normalized
                    if similarity_hash not in seen:
                        seen.add(similarity_hash)
                        unique_sentences.append(sent_stripped)

        answer = '\n'.join(unique_sentences[:12])
        answer = re.sub(r"\n\s*\n", "\n", answer)
        cleaned = answer.strip()

        if len(cleaned) < 10 and raw_answer and len(raw_answer.strip()) > 10:
            has_thinking_prefix = bool(re.match(
                r'\d*\s*(?:Thinking\s*Process|Analyze\s+the\s+Request|Let\s+me\s+think)',
                raw_answer, re.IGNORECASE,
            ))
            if has_thinking_prefix:
                logger.warning(f"_clean_answer 检测到纯thinking block且无实际答案，返回空")
                cleaned = ""
            else:
                logger.warning(f"_clean_answer 清理后过短({len(cleaned)} chars), 使用安全降级清理")
                safe = raw_answer
                for sw in ["assistant", "user:", "system:", "\n\n\n"]:
                    if sw in safe.lower():
                        idx = safe.lower().find(sw)
                        if idx > 3:
                            safe = safe[:idx]
                            break
                safe = re.sub(r'^\s*(?:Thinking\s*Process|Analyze\s+the\s+Request)\s*:\s*', '', safe, flags=re.IGNORECASE)
                safe = re.sub(r'<think\s*>.*?</think\s*>', '', safe, flags=re.DOTALL | re.IGNORECASE)
                safe = re.sub(r"\*\*", "", safe)
                safe = re.sub(r"<[^>]+>", "", safe)
                safe = re.sub(r"```\w*\s*", "", safe)
                safe = safe.lstrip('\n*-. ')
                if len(safe.strip()) > len(cleaned):
                    cleaned = safe.strip()

        return cleaned

    def invoke(self, prompt: str) -> str:
        """Direct LLM call compatible with LangChain invoke protocol.

        Unlike generate_answer(), this passes the prompt straight to the model
        without template construction or answer cleaning.

        Args:
            prompt: Fully constructed prompt text.

        Returns:
            Raw model output (unprocessed).
        """
        try:
            return self._generate(
                prompt,
                max_new_tokens=self.settings.LLM_MAX_TOKENS,
                temperature=self.settings.LLM_TEMPERATURE,
            )
        except Exception as e:
            logger.error(f"LLM invoke failed: {e}")
            raise


_llm_client_instance: Optional[LLMClient] = None
_llm_client_lock = threading.Lock()


def get_llm_client(settings) -> LLMClient:
    """Return the thread-safe LLMClient singleton via double-checked locking.

    Args:
        settings: Settings object (only used on first call).

    Returns:
        The global LLMClient instance.
    """
    global _llm_client_instance

    if _llm_client_instance is None:
        with _llm_client_lock:
            if _llm_client_instance is None:
                _llm_client_instance = LLMClient(settings)

    return _llm_client_instance
