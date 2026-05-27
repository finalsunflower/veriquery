"""
Visual indexing module for the VeriQuery ingestion layer.

Implements the ColPali (Late Interaction + Multi-vector Retrieval) architecture
for circuit diagram retrieval from PDF datasheets. Uses CLIP for fast
pre-filtering and Qwen3.5-2B VLM for deep patch-level feature extraction
with MaxSim search.

Main components:
    - CLIP_CLASS_PROMPTS: Text prompts for CLIP zero-shot image classification
    - MultiVectorEmbedding: Data container for per-image patch embeddings
    - ColPaliSearchResult: Data container for retrieval results
    - Qwen35Adapter: Wrapper for Qwen3.5-2B VLM loading and inference
    - TrueColPaliIndexer: Core indexer with CLIP filtering, embedding generation,
      MaxSim search, circuit analysis, and index persistence
    - create_visual_indexer: Factory function (singleton pattern)
    - get_visual_indexer: Lazy-accessor for the singleton instance
"""

import os
import re
import io
import contextlib
import gc
import logging
import pickle

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, AttributeError, ValueError) as e:
    import warnings
    warnings.warn(f"torch import failed: {e}, disabling torch features")
    TORCH_AVAILABLE = False
    torch = None

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import OrderedDict
from PIL import Image
import json

logger = logging.getLogger(__name__)


CLIP_CLASS_PROMPTS = {
    "circuit_diagram": [
        "Electronic circuit schematic diagram with components",
        "Circuit schematic with resistors capacitors and ICs",
        "Analog amplifier circuit with op-amp and feedback",
        "Application circuit schematic from datasheet",
        "Power supply filter and voltage regulator circuit",
    ],
    "application_circuit": [
        "Typical application circuit example schematic",
        "Reference design circuit with recommended components",
        "IC application circuit from datasheet",
    ],
    "test_circuit": [
        "Electronic test circuit schematic",
        "Evaluation board circuit diagram",
    ],
    "table": [
        "Technical data table with specifications",
        "Electrical characteristics parameter table",
        "Component specification table grid",
    ],
    "package_materials": [
        "Package materials information table",
        "Tape and reel packaging specification",
        "Reel dimension and ordering table",
    ],
    "chart": [
        "Performance graph or frequency response curve",
        "Bode plot or transfer function curve",
    ],
    "timing_diagram": [
        "Digital timing diagram with waveforms",
        "Clock and data timing signal diagram",
    ],
    "mechanical_drawing": [
        "Mechanical package dimension drawing",
        "Component outline with measurements",
    ],
    "pcb_layout": [
        "PCB land pattern footprint layout",
        "Solder pad pattern for PCB design",
    ],
    "logo": [
        "Company logo or brand symbol",
    ],
    "pinout_diagram": [
        "IC pinout configuration diagram",
        "Chip pin assignment top view",
    ],
    "block_diagram": [
        "System block diagram with modules",
        "Functional architecture diagram",
    ],
    "photo": [
        "Photograph of electronic hardware",
        "PCB board photo with components",
    ],
    "truth_table": [
        "Digital logic truth table",
        "Boolean function input output table",
    ],
    "state_diagram": [
        "State machine transition diagram",
        "Process flowchart with states",
    ],
    "other": [
        "General technical illustration",
    ]
}


@dataclass
class MultiVectorEmbedding:
    """Multi-vector embedding for a single image (ColPali architecture).

    Each image is encoded into N patch-level vectors instead of a single
    vector, preserving spatial granularity for Late Interaction retrieval.

    Attributes:
        patch_embeddings: Tensor of shape [num_patches, hidden_dim].
        attention_mask: Boolean tensor of shape [num_patches].
        image_id: Unique identifier, typically "{document_id}_page_{page_num}".
        page_num: 1-based page number.
        spatial_layout: Dict with patch-to-image position mapping.
        metadata: Optional dict with model name, dimensions, etc.
    """
    patch_embeddings: torch.Tensor
    attention_mask: torch.Tensor
    image_id: str
    page_num: int
    spatial_layout: Dict[str, Any]
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize embedding data to a JSON-compatible dictionary."""
        return {
            'patch_embeddings': self.patch_embeddings.tolist(),
            'attention_mask': self.attention_mask.tolist(),
            'image_id': self.image_id,
            'page_num': self.page_num,
            'spatial_layout': self.spatial_layout,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiVectorEmbedding':
        """Deserialize from a dictionary (used when loading persisted index)."""
        patch_data = data.get('patch_embeddings', [])
        mask_data = data.get('attention_mask', [])

        patch_embeddings = torch.tensor(patch_data) if patch_data else torch.zeros((0, 0))
        attention_mask = torch.tensor(mask_data) if mask_data else torch.zeros(0)

        return cls(
            patch_embeddings=patch_embeddings,
            attention_mask=attention_mask,
            image_id=data.get('image_id', ''),
            page_num=data.get('page_num', 0),
            spatial_layout=data.get('spatial_layout', {}),
            metadata=data.get('metadata', {})
        )


@dataclass
class ColPaliSearchResult:
    """Single retrieval result from ColPali search.

    Attributes:
        document_id: Source document identifier.
        page_num: 1-based page number.
        score: Raw MaxSim similarity score.
        confidence: Normalized confidence in [0, 1].
        metadata: Optional dict with image path, circuit type, components, etc.
        image_id: Unique image identifier.
        text: Matched text (caption for metadata search, empty for embedding search).
    """
    document_id: str
    page_num: int
    score: float
    confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    image_id: str = ""
    text: str = ""


class Qwen35Adapter:
    """Adapter for Qwen3.5-2B vision-language model.

    Supports three loading modes:
        1. Shared model: reuse an already-loaded instance from ModelManager
        2. Quantized loading: 4-bit NF4 or 8-bit INT8 via BitsAndBytesConfig
        3. Standard loading: FP16/BF16 as fallback when quantization fails
    """

    def __init__(self, model_name: str, device: str, quantize: bool = True,
                 quantization_bits: int = 4, shared_model: Dict[str, Any] = None,
                 max_image_size: int = None):
        """
        Args:
            model_name: HuggingFace model path (local or Hub name).
            device: Inference device, e.g. "cuda:0", "cpu", or "auto".
            quantize: Whether to enable quantization (saves ~75% GPU memory).
            quantization_bits: Quantization bits, 4 (NF4, recommended) or 8 (INT8).
            shared_model: Dict with 'model' and 'processor' to reuse an existing instance.
            max_image_size: Max image dimension in pixels; images exceeding this are resized.
        """
        from core.config import get_settings
        settings = get_settings()

        self.device = device
        self.model_name = model_name
        self.quantize = quantize
        self.quantization_bits = quantization_bits
        self.shared_model = shared_model
        self.max_image_size = max_image_size or settings.MAX_IMAGE_SIZE
        self.model = None
        self.processor = None
        self._load()

    def unload_model(self):
        """Unload Qwen3.5 model to free GPU memory.

        Called by ModelManager after circuit indexing completes.
        The model will be reloaded on next inference via _ensure_model_loaded().
        """
        if self.model is not None:
            import gc
            del self.model
            self.model = None
            logger.info("Qwen3.5 model unloaded from Qwen35Adapter")
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_model_loaded(self):
        """Reload the model if it was previously unloaded."""
        if self.model is None and self.model_name:
            logger.info("Qwen3.5 model not loaded, reloading on demand...")
            self._load()

    def _load(self):
        """Load Qwen3.5-2B model with priority: shared > quantized > standard."""
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

            if self.shared_model:
                logger.info("Using shared Qwen3.5 model instance")
                self.model = self.shared_model.get('model')
                self.processor = self.shared_model.get('processor')
                if self.model is None:
                    raise RuntimeError("Shared model is invalid (model=None)")
                if hasattr(self.model, 'device'):
                    self.device = str(self.model.device)
                logger.info("Shared Qwen3.5 model loaded successfully")
                return

            logger.info(f"Loading Qwen3.5 adapter: {self.model_name}")
            logger.info(f"Quantization: {self.quantize}, bits: {self.quantization_bits}")

            target_model = self.model_name if self.model_name else get_settings().VLM_MODEL

            try:
                self.processor = AutoProcessor.from_pretrained(
                    target_model, trust_remote_code=True, local_files_only=True)
            except Exception as proc_err:
                logger.warning(f"Processor local load failed ({proc_err}), trying online")
                self.processor = AutoProcessor.from_pretrained(
                    target_model, trust_remote_code=True)

            bnb_config = None
            if self.quantize and "cuda" in self.device:
                try:
                    if self.quantization_bits == 4:
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            llm_int8_enable_fp32_cpu_offload=True
                        )
                        logger.info("Qwen3.5: using 4-bit NF4 quantization (bf16 compute, CPU offload)")
                    elif self.quantization_bits == 8:
                        bnb_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_enable_fp32_cpu_offload=True
                        )
                        logger.info("Qwen3.5: using 8-bit quantization (CPU offload)")

                    if bnb_config:
                        gpu_memory_gb = self._detect_gpu_memory()
                        _cpu_mem_limit = "4GB"
                        self.model = AutoModelForImageTextToText.from_pretrained(
                            target_model,
                            quantization_config=bnb_config,
                            device_map="auto",
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            max_memory={0: f"{gpu_memory_gb}GB", "cpu": _cpu_mem_limit}
                        ).eval()
                        logger.info(f"Qwen3.5 quantized model loaded (device_map=auto, max_memory={gpu_memory_gb}GB, cpu={_cpu_mem_limit})")
                except Exception as e:
                    logger.warning(f"Qwen3.5 quantized loading failed ({e}), falling back to standard loading")

            if self.model is None:
                _dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                gpu_memory_gb = self._detect_gpu_memory()
                _cpu_mem_limit = "4GB"
                self.model = AutoModelForImageTextToText.from_pretrained(
                    target_model,
                    torch_dtype=_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: f"{gpu_memory_gb}GB", "cpu": _cpu_mem_limit}
                ).eval()
                logger.info(f"Qwen3.5 standard model loaded ({_dtype}, device_map=auto, cpu={_cpu_mem_limit})")

            try:
                from core.model_manager import model_manager
                model_manager.register_model("qwen35", self.model)
                model_manager.register_model("qwen35_processor", self.processor)
                logger.info("Qwen3.5 model registered to ModelManager")
            except Exception as e:
                logger.warning(f"Failed to register to ModelManager: {e}")

        except Exception as e:
            logger.error(f"Qwen3.5 adapter initialization failed: {e}")
            self.model = None
            self.processor = None

    def _detect_gpu_memory(self) -> float:
        """Detect available GPU memory and return a safe allocation limit in GB."""
        if not torch.cuda.is_available():
            return 2.0
        try:
            free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(0)
            free_memory = free_mem_bytes / (1024**3)
            total_memory = total_mem_bytes / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_memory_gb = max(round(free_memory * 0.85, 1), 2.0)
            logger.info(f"GPU total={total_memory:.1f}GB, used={allocated_memory:.1f}GB, free={free_memory:.1f}GB, alloc={gpu_memory_gb}GB")
            return gpu_memory_gb
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info, using conservative value: {e}")
            return 2.0

    def _prepare_inputs(self, image: Image.Image, text_prompt: str) -> Dict[str, Any]:
        """Preprocess image and text into model input tensors.

        Args:
            image: PIL image object.
            text_prompt: Text prompt guiding the model's attention.

        Returns:
            Dict of input tensors (input_ids, attention_mask, pixel_values, etc.).
        """
        original_size = image.size
        if max(original_size) > self.max_image_size:
            ratio = self.max_image_size / max(original_size)
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            logger.debug(f"Image resized: {original_size} -> {new_size}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _pvi = _get_process_vision_info()
        image_inputs, video_inputs = _pvi(messages)
        if image_inputs is None:
            image_inputs = []
        if video_inputs is None:
            video_inputs = []

        processor_kwargs = {
            "text": [text],
            "images": image_inputs if image_inputs else None,
            "padding": True,
            "return_tensors": "pt"
        }
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
        inputs = self.processor(**processor_kwargs)
        return inputs

    def extract_patch_features(self, image: Image.Image) -> Optional[torch.Tensor]:
        """Extract patch-level feature vectors from an image (ColPali core step).

        Returns:
            Tensor of shape [num_patches, hidden_dim], or None on failure.
        """
        self._ensure_model_loaded()
        if self.model is None:
            logger.error("Qwen3.5 model not loaded, cannot extract patch features")
            return None
        try:
            gpu_available = False
            if torch.cuda.is_available():
                try:
                    free_memory, total_memory = torch.cuda.mem_get_info(0)
                    free_memory_gb = free_memory / (1024**3)
                    if free_memory_gb < 0.5:
                        torch.cuda.empty_cache()
                        gc.collect()
                        free_memory, _ = torch.cuda.mem_get_info(0)
                        free_memory_gb = free_memory / (1024**3)
                    gpu_available = free_memory_gb >= 0.5
                    if not gpu_available:
                        logger.warning(f"Insufficient GPU memory ({free_memory_gb:.2f}GB), trying CPU fallback")
                except Exception as e:
                    logger.warning(f"GPU memory check failed: {e}, trying CPU fallback")

            original_device = self.device
            if not gpu_available and "cuda" in self.device:
                logger.info("Insufficient GPU memory, switching to CPU inference")
                try:
                    if self.model is not None:
                        self.model = self.model.to("cpu")
                        self.device = "cpu"
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Failed to move model to CPU: {e}")
                    return None

            circuit_prompt = "Analyze this electronic circuit diagram. Identify components, topology, and signal flow."
            inputs = self._prepare_inputs(image, circuit_prompt)
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            with torch.no_grad():
                _amp_ctx = (
                    torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=False)
                    if "cuda" in self.device and torch.cuda.is_available()
                    else contextlib.nullcontext()
                )
                with _amp_ctx:
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        output_attentions=False,
                        return_dict=True
                    )

            _hs = outputs.get('hidden_states') if isinstance(outputs, dict) else getattr(outputs, 'hidden_states', None)

            if _hs is not None and len(_hs) > 0:
                patch_embeddings = _hs[-1].squeeze(0)
            else:
                _lhs = outputs.get('last_hidden_state') if isinstance(outputs, dict) else getattr(outputs, 'last_hidden_state', None)
                if _lhs is not None:
                    logger.warning("hidden_states is empty, falling back to last_hidden_state")
                    patch_embeddings = _lhs.squeeze(0)
                else:
                    _keys = list(outputs.keys()) if isinstance(outputs, dict) else [k for k in dir(outputs) if not k.startswith('_')]
                    raise RuntimeError(f"Model output missing hidden_states/last_hidden_state, keys: {_keys[:10]}")

            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not gpu_available and "cuda" not in self.device and torch.cuda.is_available():
                try:
                    free_memory, _ = torch.cuda.mem_get_info(0)
                    if free_memory / (1024**3) >= 1.5:
                        self.model = self.model.to(original_device)
                        self.device = original_device
                        logger.info(f"Inference complete, model moved back to {original_device}")
                except Exception as e:
                    logger.warning(f"Failed to move model back to GPU: {e}")

            return patch_embeddings

        except Exception as e:
            logger.error(f"Qwen feature extraction failed: {e}")
            import traceback
            logger.error(f"Detailed error:\n{traceback.format_exc()}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    def generate_text(self, image: Image.Image, prompt: str, max_new_tokens: int = 512) -> Optional[str]:
        """Generate text description from an image using the VLM.

        Args:
            image: PIL image object.
            prompt: Text prompt for generation.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Generated text string, or None on failure.
        """
        self._ensure_model_loaded()
        if self.model is None or self.processor is None:
            logger.error("Qwen3.5 model not loaded, cannot generate text")
            return None
        try:
            inputs = self._prepare_inputs(image, prompt)
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            with torch.no_grad():
                _amp_ctx = (
                    torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=False)
                    if "cuda" in self.device and torch.cuda.is_available()
                    else contextlib.nullcontext()
                )
                with _amp_ctx:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                    )

            input_len = inputs['input_ids'].shape[1]
            output_ids = generated_ids[:, input_len:]
            response = self.processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0].strip()

            del inputs, generated_ids, output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return response

        except Exception as e:
            logger.error(f"Qwen text generation failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None


try:
    from qwen_vl_utils import process_vision_info as _qwen_process_vision_info

    def process_vision_info(messages: List[Dict]) -> Tuple[List, None]:
        try:
            result = _qwen_process_vision_info(messages)
            if result is not None:
                image_inputs, video_inputs = result
                return image_inputs if image_inputs else [], video_inputs
            return [], None
        except Exception as e:
            logger.debug(f"qwen_vl_utils.process_vision_info failed, using builtin: {e}")
            return _builtin_process_vision_info(messages)

    logger.info("Using qwen_vl_utils official process_vision_info")
except ImportError:
    process_vision_info = None
    logger.info("qwen_vl_utils not installed, will use builtin process_vision_info")


def _builtin_process_vision_info(messages: List[Dict]) -> Tuple[List, None]:
    """Fallback implementation of process_vision_info when qwen_vl_utils is unavailable."""
    image_inputs = []
    for message in messages:
        for content in message.get("content", []):
            if content.get("type") == "image":
                img = content.get("image")
                if img is not None:
                    if isinstance(img, str):
                        try:
                            from PIL import Image as PILImage
                            if img.startswith(("http://", "https://")):
                                import requests
                                resp = requests.get(img, timeout=10)
                                img = PILImage.open(io.BytesIO(resp.content))
                            elif img.startswith("data:"):
                                import base64
                                _, base64_data = img.split(",", 1)
                                img = PILImage.open(io.BytesIO(base64.b64decode(base64_data)))
                            else:
                                img = PILImage.open(img)
                        except Exception as e:
                            logger.warning(f"Failed to load image {img}: {e}")
                            continue
                    image_inputs.append(img)
            elif content.get("type") == "image_url":
                url = content.get("image_url", {})
                if isinstance(url, dict):
                    url = url.get("url", "")
                if url:
                    try:
                        from PIL import Image as PILImage
                        if url.startswith(("http://", "https://")):
                            import requests
                            resp = requests.get(url, timeout=10)
                            image_inputs.append(PILImage.open(io.BytesIO(resp.content)))
                        elif url.startswith("data:"):
                            import base64
                            _, base64_data = url.split(",", 1)
                            image_inputs.append(PILImage.open(io.BytesIO(base64.b64decode(base64_data))))
                        else:
                            image_inputs.append(PILImage.open(url))
                    except Exception as e:
                        logger.warning(f"Failed to load image URL {url}: {e}")
    return image_inputs, None


def _get_process_vision_info():
    if process_vision_info is not None:
        return process_vision_info
    return _builtin_process_vision_info


class TrueColPaliIndexer:
    """ColPali-based visual indexer for circuit diagram retrieval.

    Combines CLIP zero-shot classification for fast pre-filtering with
    Qwen3.5-2B VLM for deep patch-level embedding extraction. Retrieval
    uses the MaxSim (Late Interaction) scoring to match query and document
    patch embeddings at fine granularity.

    Attributes:
        _CIRCUIT_TYPE_KEYWORDS: Class-level keyword map for circuit type detection.
    """

    _CIRCUIT_TYPE_KEYWORDS = {
        "amplifier": ["amplifier", "op-amp", "运放", "增益", "gain", "反馈", "feedback"],
        "filter": ["filter", "滤波", "低通", "高通", "带通", "LPF", "HPF", "BPF"],
        "power": ["power", "电源", "稳压", "regulator", "LDO", "DC-DC", "供电"],
        "oscillator": ["oscillator", "振荡", "晶振", "crystal", "时钟", "clock"],
        "driver": ["driver", "驱动", "LED驱动", "电机驱动", "H桥", "PWM"],
        "comparator": ["comparator", "比较器", "阈值", "threshold"],
        "timer": ["timer", "定时", "555", "延时", "delay"],
        "rectifier": ["rectifier", "整流", "桥式", "半波", "全波"],
        "protection": ["protection", "保护", "ESD", "TVS", "过流", "过压"],
        "interface": ["interface", "接口", "UART", "SPI", "I2C", "USB", "CAN"],
        "reset": ["reset", "复位", "POR", "看门狗", "watchdog"],
        "logic": ["logic", "逻辑", "门电路", "触发器", "flip-flop"],
    }

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu",
                 quantize: bool = True, quantization_bits: int = 4,
                 enable_clip_prefilter: bool = True,
                 clip_model_name: Optional[str] = None,
                 max_cache_size: int = 100,
                 index_dir: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model path for Qwen3.5-2B VLM.
                None disables VLM features (metadata-only mode).
            device: Inference device, e.g. "cuda:0" or "cpu".
            quantize: Whether to quantize the VLM model.
            quantization_bits: 4 (NF4) or 8 (INT8).
            enable_clip_prefilter: Whether to load CLIP for image classification.
            clip_model_name: HuggingFace model path for CLIP.
            max_cache_size: Maximum LRU cache entries for embedding vectors.
            index_dir: Directory for index persistence. Defaults to data/visual_index.
        """
        from core.config import get_settings
        settings = get_settings()

        self.model_name = model_name
        self.device = device
        self.quantize = quantize
        self.quantization_bits = quantization_bits
        self.enable_clip_prefilter = enable_clip_prefilter
        self.clip_model_name = clip_model_name or settings.CLIP_MODEL
        self.max_cache_size = max_cache_size

        self._qwen_adapter: Optional[Qwen35Adapter] = None
        self._qwen_loaded = False
        self._colpali_model = None

        self._clip_model = None
        self._clip_processor = None
        self._clip_loaded = False

        self._embeddings: OrderedDict[str, MultiVectorEmbedding] = OrderedDict()
        self._document_index: List[Dict[str, Any]] = []
        self._text_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        self._index_dir = Path(index_dir) if index_dir else Path(settings.DATA_DIR) / "visual_index"
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._circuit_images_dir = Path(settings.DATA_DIR) / "circuit_images"
        self._circuit_images_dir.mkdir(parents=True, exist_ok=True)

        self._load_persisted_index()

        logger.info(f"TrueColPaliIndexer initialized (device={device}, clip={enable_clip_prefilter}, vlm={model_name is not None})")

    def _load_persisted_index(self):
        """Load previously persisted circuit index and embeddings from disk."""
        index_path = self._index_dir / "circuit_index.json"
        embeddings_path = self._index_dir / "embeddings_index.pkl"

        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._document_index = data.get('document_index', [])
                logger.info(f"Loaded circuit index: {len(self._document_index)} entries from {index_path}")
            except Exception as e:
                logger.warning(f"Failed to load circuit index: {e}")
                self._document_index = []

        if embeddings_path.exists():
            try:
                with open(embeddings_path, 'rb') as f:
                    emb_data = pickle.load(f)
                if isinstance(emb_data, dict):
                    for image_id, emb_dict in emb_data.items():
                        try:
                            self._embeddings[image_id] = MultiVectorEmbedding.from_dict(emb_dict)
                        except Exception as e:
                            logger.warning(f"Failed to deserialize embedding {image_id}: {e}")
                logger.info(f"Loaded embeddings: {len(self._embeddings)} entries from {embeddings_path}")
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")

    def flush_index_to_disk(self):
        """Persist current circuit index and embeddings to disk."""
        self._index_dir.mkdir(parents=True, exist_ok=True)

        index_path = self._index_dir / "circuit_index.json"
        try:
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump({'document_index': self._document_index}, f, ensure_ascii=False, indent=2)
            logger.info(f"Circuit index flushed: {len(self._document_index)} entries -> {index_path}")
        except Exception as e:
            logger.error(f"Failed to flush circuit index: {e}")

        embeddings_path = self._index_dir / "embeddings_index.pkl"
        try:
            emb_data = {}
            for image_id, emb in self._embeddings.items():
                try:
                    emb_data[image_id] = emb.to_dict()
                except Exception as e:
                    logger.warning(f"Failed to serialize embedding {image_id}: {e}")
            with open(embeddings_path, 'wb') as f:
                pickle.dump(emb_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Embeddings flushed: {len(emb_data)} entries -> {embeddings_path}")
        except Exception as e:
            logger.error(f"Failed to flush embeddings: {e}")

    def _load_model(self, model_type: str = "qwen"):
        """Load a specific model on demand.

        Args:
            model_type: "qwen" for VLM, "clip" for CLIP model.
        """
        if model_type == "qwen" and not self._qwen_loaded:
            if self.model_name is None:
                logger.warning("VLM model_name is None, skipping Qwen load")
                return
            try:
                shared_model = None
                try:
                    from core.model_manager import model_manager
                    existing_model = model_manager.get_model("qwen35")
                    existing_processor = model_manager.get_model("qwen35_processor")
                    if existing_model is not None:
                        shared_model = {'model': existing_model, 'processor': existing_processor}
                        logger.info("Reusing shared Qwen3.5 model from ModelManager")
                except Exception:
                    pass

                self._qwen_adapter = Qwen35Adapter(
                    model_name=self.model_name,
                    device=self.device,
                    quantize=self.quantize,
                    quantization_bits=self.quantization_bits,
                    shared_model=shared_model,
                )
                self._qwen_loaded = True
                self._colpali_model = self._qwen_adapter
                logger.info("Qwen3.5 VLM loaded on demand")
            except Exception as e:
                logger.error(f"Failed to load Qwen3.5 VLM: {e}")

        elif model_type == "clip" and not self._clip_loaded:
            try:
                from transformers import CLIPModel, CLIPProcessor
                self._clip_processor = CLIPProcessor.from_pretrained(
                    self.clip_model_name, trust_remote_code=True, local_files_only=True)
                self._clip_model = CLIPModel.from_pretrained(
                    self.clip_model_name, trust_remote_code=True, local_files_only=True)
                if "cuda" in self.device and torch.cuda.is_available():
                    self._clip_model = self._clip_model.to(self.device)
                self._clip_model.eval()
                self._clip_loaded = True
                logger.info(f"CLIP model loaded: {self.clip_model_name}")
            except Exception as e:
                logger.warning(f"CLIP local load failed ({e}), trying online")
                try:
                    from transformers import CLIPModel, CLIPProcessor
                    self._clip_processor = CLIPProcessor.from_pretrained(
                        self.clip_model_name, trust_remote_code=True)
                    self._clip_model = CLIPModel.from_pretrained(
                        self.clip_model_name, trust_remote_code=True)
                    if "cuda" in self.device and torch.cuda.is_available():
                        self._clip_model = self._clip_model.to(self.device)
                    self._clip_model.eval()
                    self._clip_loaded = True
                    logger.info(f"CLIP model loaded (online): {self.clip_model_name}")
                except Exception as e2:
                    logger.error(f"CLIP model load failed: {e2}")

    def unload_model(self):
        """Unload VLM model to free GPU memory."""
        if self._qwen_adapter is not None:
            self._qwen_adapter.unload_model()
            self._qwen_loaded = False
            self._colpali_model = None
            logger.info("VLM model unloaded from TrueColPaliIndexer")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def classify_image_with_clip(self, image: Image.Image) -> Tuple[str, float]:
        """Classify an image using CLIP zero-shot classification.

        Args:
            image: PIL image to classify.

        Returns:
            Tuple of (predicted_class, confidence_score).
        """
        if not self._clip_loaded:
            self._load_model("clip")
        if self._clip_model is None or self._clip_processor is None:
            return "other", 0.0

        try:
            all_prompts = []
            class_names = []
            for class_name, prompts in CLIP_CLASS_PROMPTS.items():
                for prompt in prompts:
                    all_prompts.append(prompt)
                    class_names.append(class_name)

            inputs = self._clip_processor(
                text=all_prompts, images=image, return_tensors="pt", padding=True
            )
            if "cuda" in self.device and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1)

            class_scores = {}
            for i, class_name in enumerate(class_names):
                score = probs[0, i].item()
                class_scores[class_name] = max(class_scores.get(class_name, 0.0), score)

            best_class = max(class_scores, key=class_scores.get)
            best_score = class_scores[best_class]

            del inputs, outputs, logits, probs
            return best_class, best_score

        except Exception as e:
            logger.error(f"CLIP classification failed: {e}")
            return "other", 0.0

    def _extract_embeddings(self, image: Image.Image) -> Optional[MultiVectorEmbedding]:
        """Extract multi-vector patch embeddings from an image.

        Args:
            image: PIL image object.

        Returns:
            MultiVectorEmbedding instance, or None on failure.
        """
        if not self._qwen_loaded:
            self._load_model("qwen")
        if self._qwen_adapter is None:
            logger.error("VLM adapter not available for embedding extraction")
            return None

        patch_features = self._qwen_adapter.extract_patch_features(image)
        if patch_features is None:
            return None

        num_patches = patch_features.shape[0]
        attention_mask = torch.ones(num_patches, dtype=torch.bool)

        spatial_layout = {
            "width": image.width,
            "height": image.height,
            "num_patches": num_patches,
        }

        return MultiVectorEmbedding(
            patch_embeddings=patch_features,
            attention_mask=attention_mask,
            image_id="",
            page_num=0,
            spatial_layout=spatial_layout,
            metadata={"model": "qwen3.5-2b", "hidden_dim": patch_features.shape[1]}
        )

    def index_document(self, document_path: Path, metadata: Dict[str, Any] = None,
                       preloaded_images: List[Image.Image] = None,
                       page_numbers: List[int] = None) -> Dict[str, Any]:
        """Index a document's circuit diagrams with ColPali embeddings.

        Extracts images from the PDF (or uses preloaded images), runs CLIP
        pre-filtering, generates patch-level embeddings via Qwen3.5-2B, and
        stores them for MaxSim retrieval.

        Args:
            document_path: Path to the PDF document.
            metadata: Dict with document_id, file_path, pages_count, etc.
            preloaded_images: Pre-extracted PIL images (avoids re-parsing PDF).
            page_numbers: Optional 1-based page numbers corresponding to preloaded_images.

        Returns:
            Dict with status, document_id, indexed_count, and embedding_ids.
        """
        metadata = metadata or {}
        document_id = metadata.get('document_id', Path(document_path).stem)
        filename = metadata.get('filename', Path(document_path).name)

        try:
            if preloaded_images:
                images = preloaded_images
                pages = page_numbers or list(range(1, len(images) + 1))
            else:
                images, pages = self._extract_images_from_pdf(document_path)

            if not images:
                logger.warning(f"No images extracted from {filename}")
                return {"status": "no_images", "document_id": document_id, "indexed_count": 0}

            if not self._qwen_loaded:
                self._load_model("qwen")

            indexed_count = 0
            embedding_ids = []

            for i, (image, page_num) in enumerate(zip(images, pages)):
                try:
                    is_circuit = True
                    clip_type = None
                    clip_confidence = 0.0

                    if self.enable_clip_prefilter and self._clip_loaded:
                        clip_type, clip_confidence = self.classify_image_with_clip(image)
                        circuit_types = {"circuit_diagram", "application_circuit", "test_circuit", "block_diagram"}
                        is_circuit = clip_type in circuit_types or clip_confidence >= 0.15

                    if not is_circuit:
                        logger.debug(f"Skipping non-circuit page {page_num} (type={clip_type})")
                        continue

                    image_id = f"{document_id}_page_{page_num}"

                    image_save_dir = self._circuit_images_dir / document_id
                    image_save_dir.mkdir(parents=True, exist_ok=True)
                    image_save_path = image_save_dir / f"page{page_num}_img0.png"
                    image.save(str(image_save_path), "PNG")

                    embedding = self._extract_embeddings(image)
                    if embedding is None:
                        logger.warning(f"Embedding extraction failed for page {page_num}")
                        continue

                    embedding.image_id = image_id
                    embedding.page_num = page_num

                    self._embeddings[image_id] = embedding
                    if len(self._embeddings) > self.max_cache_size:
                        self._embeddings.popitem(last=False)

                    index_entry = {
                        "document_id": document_id,
                        "filename": filename,
                        "page": page_num,
                        "image_id": image_id,
                        "image_path": str(image_save_path),
                        "caption": "",
                        "circuit_type": [],
                        "components": [],
                        "is_circuit": True,
                        "num_patches": embedding.patch_embeddings.shape[0],
                        "spatial_layout": embedding.spatial_layout,
                    }

                    if clip_type:
                        index_entry["clip_type"] = clip_type
                        index_entry["clip_confidence"] = clip_confidence

                    self._document_index.append(index_entry)
                    embedding_ids.append(image_id)
                    indexed_count += 1

                    logger.debug(f"Indexed page {page_num}: {image_id} ({embedding.patch_embeddings.shape[0]} patches)")

                    del embedding
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.warning(f"Failed to index page {page_num}: {e}")
                    continue

            self.flush_index_to_disk()

            logger.info(f"Document indexed: {filename}, {indexed_count}/{len(images)} pages with embeddings")
            return {
                "status": "success",
                "document_id": document_id,
                "indexed_count": indexed_count,
                "total_images": len(images),
                "embedding_ids": embedding_ids,
            }

        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return {
                "status": "failed",
                "document_id": document_id,
                "error": str(e),
                "indexed_count": 0,
            }

    def _extract_images_from_pdf(self, document_path) -> Tuple[List[Image.Image], List[int]]:
        """Extract page images from a PDF document.

        Args:
            document_path: Path to the PDF file.

        Returns:
            Tuple of (images_list, page_numbers_list).
        """
        images = []
        page_numbers = []
        try:
            import fitz
            doc = fitz.open(str(document_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=200)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                images.append(image)
                page_numbers.append(page_num + 1)
            doc.close()
        except ImportError:
            logger.warning("PyMuPDF not available, cannot extract images from PDF")
        except Exception as e:
            logger.error(f"PDF image extraction failed: {e}")
        return images, page_numbers

    def search_documents(self, query: str, top_k: int = 10,
                         score_threshold: float = 0.1,
                         document_ids: Optional[List[str]] = None) -> List[ColPaliSearchResult]:
        """Search indexed circuit diagrams using MaxSim (Late Interaction).

        Encodes the query into patch-level embeddings via Qwen3.5, then
        computes MaxSim scores against all indexed document embeddings.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            score_threshold: Minimum similarity score for results.
            document_ids: Optional filter to restrict search to specific documents.

        Returns:
            List of ColPaliSearchResult sorted by score descending.
        """
        if not self._embeddings:
            logger.warning("No embeddings indexed, returning empty results")
            return []

        if not self._qwen_loaded:
            self._load_model("qwen")

        query_embedding = self._get_query_embedding(query)
        if query_embedding is None:
            logger.warning("Failed to encode query, falling back to metadata search")
            return self._search_with_metadata(query, top_k, score_threshold, document_ids)

        results = []
        for image_id, doc_embedding in self._embeddings.items():
            if document_ids:
                entry = self._find_index_entry(image_id)
                if entry and entry.get('document_id') not in document_ids:
                    continue

            score = self._compute_maxsim(query_embedding, doc_embedding)
            if score >= score_threshold:
                entry = self._find_index_entry(image_id)
                if entry:
                    results.append(ColPaliSearchResult(
                        document_id=entry.get('document_id', ''),
                        page_num=entry.get('page', 0),
                        score=score,
                        confidence=min(score, 1.0),
                        metadata=entry,
                        image_id=image_id,
                        text=entry.get('caption', ''),
                    ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _search_with_metadata(self, query: str, top_k: int = 10,
                              score_threshold: float = 0.1,
                              document_ids: Optional[List[str]] = None) -> List[ColPaliSearchResult]:
        """Fallback search using metadata (caption, circuit_type, components) matching.

        Used when VLM is unavailable or embedding search returns empty.

        Args:
            query: Search query string.
            top_k: Maximum number of results.
            score_threshold: Minimum relevance score.
            document_ids: Optional document filter.

        Returns:
            List of ColPaliSearchResult sorted by score descending.
        """
        query_lower = query.lower()
        query_terms = set(re.findall(r'[a-zA-Z0-9]+', query_lower))
        query_terms.update(re.findall(r'[\u4e00-\u9fff]{2,}', query_lower))

        results = []
        for entry in self._document_index:
            if document_ids and entry.get('document_id') not in document_ids:
                continue

            score = self._compute_metadata_score(query_lower, query_terms, entry)
            if score >= score_threshold:
                results.append(ColPaliSearchResult(
                    document_id=entry.get('document_id', ''),
                    page_num=entry.get('page', 0),
                    score=score,
                    confidence=min(score, 1.0),
                    metadata=entry,
                    image_id=entry.get('image_id', ''),
                    text=entry.get('caption', ''),
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _compute_metadata_score(self, query_lower: str, query_terms: set,
                                entry: Dict[str, Any]) -> float:
        """Compute a relevance score between a query and an index entry's metadata.

        Args:
            query_lower: Lowercase query string.
            query_terms: Set of extracted query terms.
            entry: Index entry dict with caption, circuit_type, components, etc.

        Returns:
            Relevance score in [0, 1].
        """
        if not query_terms:
            return 0.0

        score = 0.0

        caption = entry.get('caption', '').lower()
        if caption:
            matched = sum(1 for t in query_terms if t in caption)
            score += min(matched / max(len(query_terms), 1), 1.0) * 0.4

        circuit_types = entry.get('circuit_type', [])
        if isinstance(circuit_types, str):
            try:
                circuit_types = json.loads(circuit_types)
            except (json.JSONDecodeError, TypeError):
                circuit_types = []
        if circuit_types:
            type_text = ' '.join(str(t).lower() for t in circuit_types if t)
            matched = sum(1 for t in query_terms if t in type_text)
            score += min(matched / max(len(query_terms), 1), 1.0) * 0.3

        components = entry.get('components', [])
        if isinstance(components, str):
            try:
                components = json.loads(components)
            except (json.JSONDecodeError, TypeError):
                components = []
        if components:
            comp_text = ' '.join(str(c).lower() for c in components if c)
            matched = sum(1 for t in query_terms if t in comp_text)
            score += min(matched / max(len(query_terms), 1), 1.0) * 0.2

        filename = entry.get('filename', '').lower().replace('.pdf', '')
        if filename:
            matched = sum(1 for t in query_terms if t in filename and len(t) >= 2)
            score += min(matched / max(len(query_terms), 1), 1.0) * 0.1

        return min(score, 1.0)

    def _get_query_embedding(self, query: str) -> Optional[torch.Tensor]:
        """Encode a text query into patch-level embeddings for MaxSim search.

        Uses LRU cache to avoid redundant encoding of repeated queries.

        Args:
            query: Search query string.

        Returns:
            Tensor of shape [num_patches, hidden_dim], or None on failure.
        """
        if query in self._text_cache:
            emb = self._text_cache.pop(query)
            self._text_cache[query] = emb
            return emb

        if self._qwen_adapter is None:
            return None

        try:
            query_image = self._render_query_image(query)
            embedding = self._qwen_adapter.extract_patch_features(query_image)
            if embedding is not None:
                self._text_cache[query] = embedding
                if len(self._text_cache) > self.max_cache_size:
                    self._text_cache.popitem(last=False)
            return embedding
        except Exception as e:
            logger.error(f"Query encoding failed: {e}")
            return None

    def _render_query_image(self, query: str) -> Image.Image:
        """Render a text query into a synthetic image for VLM encoding.

        Since ColPali uses vision encoders, text queries must be rendered
        as images for embedding extraction.

        Args:
            query: Text query string.

        Returns:
            PIL Image containing the rendered query text.
        """
        width, height = 800, 200
        img = Image.new('RGB', (width, height), color='white')

        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("arial.ttf", size=24)
            except (IOError, OSError):
                font = ImageFont.load_default()

            text = query[:200]
            draw.text((20, 80), text, fill='black', font=font)
        except Exception:
            pass

        return img

    def _compute_maxsim(self, query_embedding: torch.Tensor,
                        doc_embedding: MultiVectorEmbedding) -> float:
        """Compute MaxSim score between query and document patch embeddings.

        MaxSim (Late Interaction): for each query patch, find the maximum
        similarity with any document patch, then average across all query patches.

        Args:
            query_embedding: Tensor of shape [num_query_patches, hidden_dim].
            doc_embedding: MultiVectorEmbedding with document patch embeddings.

        Returns:
            MaxSim similarity score (scalar).
        """
        try:
            doc_patches = doc_embedding.patch_embeddings
            if doc_patches.dim() == 1:
                doc_patches = doc_patches.unsqueeze(0)

            query_norm = F.normalize(query_embedding.float(), p=2, dim=-1)
            doc_norm = F.normalize(doc_patches.float(), p=2, dim=-1)

            similarity = torch.mm(query_norm, doc_norm.t())
            max_scores, _ = similarity.max(dim=1)
            score = max_scores.mean().item()

            return score

        except Exception as e:
            logger.error(f"MaxSim computation failed: {e}")
            return 0.0

    def _find_index_entry(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Find an index entry by image_id.

        Args:
            image_id: Unique image identifier.

        Returns:
            Index entry dict, or None if not found.
        """
        for entry in self._document_index:
            if entry.get('image_id') == image_id:
                return entry
        return None

    def update_page_metadata(self, document_id: str, page_num: int,
                             metadata_updates: Dict[str, Any]):
        """Update metadata for a specific page in the index.

        Args:
            document_id: Document identifier.
            page_num: 1-based page number.
            metadata_updates: Dict of metadata fields to update.
        """
        for entry in self._document_index:
            if entry.get('document_id') == document_id and entry.get('page') == page_num:
                entry.update(metadata_updates)
                logger.debug(f"Updated metadata for {document_id} page {page_num}")
                return
        logger.warning(f"No index entry found for {document_id} page {page_num}")

    def remove_document_index(self, document_id: str) -> int:
        """Remove all index entries and embeddings for a document.

        Args:
            document_id: Document identifier to remove.

        Returns:
            Number of entries removed.
        """
        removed = 0
        new_index = []
        for entry in self._document_index:
            if entry.get('document_id') == document_id:
                removed += 1
            else:
                new_index.append(entry)
        self._document_index = new_index

        removed_count = 0
        keys_to_remove = [
            key for key, emb in self._embeddings.items()
            if key.startswith(f"{document_id}_page_")
        ]
        for key in keys_to_remove:
            del self._embeddings[key]
            removed_count += 1

        removed = max(removed, removed_count)
        if removed > 0:
            logger.info(f"Removed {removed} index entries for document {document_id}")
        return removed

    def analyze_circuit_image(self, image_path: str, annotation_type: str = "auto",
                              clip_type: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a circuit diagram image using the VLM.

        Generates a structured analysis including circuit type, components,
        caption, and confidence score.

        Args:
            image_path: Path to the circuit image file.
            annotation_type: Analysis mode ("auto", "detailed", "brief").
            clip_type: Pre-detected CLIP classification type (optional).

        Returns:
            Dict with success, is_circuit, and analysis sub-dict.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            return {"success": False, "is_circuit": False, "error": str(e)}

        if not self._qwen_loaded:
            self._load_model("qwen")
        if self._qwen_adapter is None:
            return {"success": False, "is_circuit": False, "error": "VLM not available"}

        try:
            if annotation_type == "brief":
                prompt = (
                    "Is this an electronic circuit diagram? "
                    "Answer YES or NO, then briefly describe the circuit type."
                )
                max_tokens = 128
            elif annotation_type == "detailed":
                prompt = (
                    "Analyze this electronic circuit diagram in detail. "
                    "Identify: 1) Circuit type (amplifier, filter, power supply, etc.) "
                    "2) Key components (ICs, resistors, capacitors, etc.) "
                    "3) Circuit topology and signal flow "
                    "4) Application scenario"
                )
                max_tokens = 512
            else:
                prompt = (
                    "Analyze this electronic circuit diagram. "
                    "Provide: 1) Circuit type 2) Key components "
                    "3) Brief description of functionality"
                )
                max_tokens = 256

            response = self._qwen_adapter.generate_text(image, prompt, max_new_tokens=max_tokens)
            if response is None:
                return {"success": False, "is_circuit": False, "error": "VLM generation failed"}

            is_circuit = self._detect_circuit_from_response(response, clip_type)
            circuit_type = self._extract_circuit_type(response, clip_type)
            components = self._extract_components(response)
            figure_label = self._extract_figure_label(response)

            confidence = 0.7 if is_circuit else 0.3
            if clip_type and clip_type in ("circuit_diagram", "application_circuit", "test_circuit"):
                confidence = min(confidence + 0.15, 1.0)

            return {
                "success": True,
                "is_circuit": is_circuit,
                "analysis": {
                    "circuit_type": circuit_type,
                    "components": components,
                    "raw_response": response,
                    "figure_label": figure_label,
                    "overall_confidence": confidence,
                }
            }

        except Exception as e:
            logger.error(f"Circuit analysis failed: {e}")
            return {"success": False, "is_circuit": False, "error": str(e)}

    def _detect_circuit_from_response(self, response: str,
                                      clip_type: Optional[str] = None) -> bool:
        """Determine whether the VLM response indicates a circuit diagram.

        Args:
            response: VLM-generated text response.
            clip_type: Optional CLIP classification type.

        Returns:
            True if the image is likely a circuit diagram.
        """
        if clip_type and clip_type in ("circuit_diagram", "application_circuit", "test_circuit", "block_diagram"):
            return True

        circuit_keywords = [
            "circuit", "schematic", "amplifier", "filter", "regulator",
            "oscillator", "op-amp", "transistor", "capacitor", "resistor",
            "电路", "原理图", "放大器", "滤波器", "稳压器", "振荡器",
        ]
        response_lower = response.lower()
        matches = sum(1 for kw in circuit_keywords if kw in response_lower)
        return matches >= 2

    def _extract_circuit_type(self, response: str,
                              clip_type: Optional[str] = None) -> List[str]:
        """Extract circuit type categories from VLM response.

        Args:
            response: VLM-generated text.
            clip_type: Optional CLIP classification hint.

        Returns:
            List of detected circuit type strings.
        """
        types = []
        if clip_type and clip_type not in ("other", "logo", "photo"):
            types.append(clip_type)

        response_lower = response.lower()
        for type_name, keywords in self._CIRCUIT_TYPE_KEYWORDS.items():
            if any(kw in response_lower for kw in keywords):
                if type_name not in types:
                    types.append(type_name)

        return types if types else ["unknown"]

    def _extract_components(self, response: str) -> List[str]:
        """Extract component names from VLM response.

        Args:
            response: VLM-generated text.

        Returns:
            List of detected component strings.
        """
        components = []
        component_patterns = [
            r'\b([A-Z]{2,}\d{2,}[A-Z0-9]*)\b',
            r'\b(LM\d{3,}|NE\d{3,}|TL\d{3,}|AD\d{3,}|OPA\d{3,})\b',
            r'\b(\d{1,4}[kKmM]?[Ωω])\b',
            r'\b(\d{1,4}[pPnNuUμmM]?[Ff])\b',
        ]
        for pattern in component_patterns:
            matches = re.findall(pattern, response)
            components.extend(matches[:5])

        return list(set(components))[:10]

    def _extract_figure_label(self, response: str) -> str:
        """Extract figure label (e.g. 'Figure 3') from VLM response.

        Args:
            response: VLM-generated text.

        Returns:
            Figure label string, or empty string if not found.
        """
        match = re.search(r'[Ff]igure\s*(\d+)', response)
        if match:
            return f"Figure {match.group(1)}"
        match = re.search(r'[图图]\s*(\d+)', response)
        if match:
            return f"图 {match.group(1)}"
        return ""


_visual_indexer_instance: Optional[TrueColPaliIndexer] = None


def create_visual_indexer(model_name: Optional[str] = None,
                          device: Optional[str] = None,
                          quantize: bool = True,
                          quantization_bits: int = 4,
                          enable_clip_prefilter: bool = True,
                          **kwargs) -> TrueColPaliIndexer:
    """Factory function for creating a TrueColPaliIndexer (singleton).

    Args:
        model_name: HuggingFace model path for Qwen3.5-2B VLM.
        device: Inference device. Auto-detected if None.
        quantize: Whether to quantize the VLM model.
        quantization_bits: 4 (NF4) or 8 (INT8).
        enable_clip_prefilter: Whether to enable CLIP pre-filtering.
        **kwargs: Additional keyword arguments passed to TrueColPaliIndexer.

    Returns:
        Singleton TrueColPaliIndexer instance.
    """
    global _visual_indexer_instance
    if _visual_indexer_instance is not None:
        return _visual_indexer_instance

    from core.config import get_settings
    settings = get_settings()

    if device is None:
        device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    if model_name is None:
        model_name = settings.VLM_MODEL

    _visual_indexer_instance = TrueColPaliIndexer(
        model_name=model_name,
        device=device,
        quantize=quantize,
        quantization_bits=quantization_bits,
        enable_clip_prefilter=enable_clip_prefilter,
        clip_model_name=settings.CLIP_MODEL,
        **kwargs,
    )

    return _visual_indexer_instance


def get_visual_indexer() -> Optional[TrueColPaliIndexer]:
    """Get the singleton visual indexer instance.

    Returns the existing instance without re-initialization. Use
    create_visual_indexer() for first-time creation.

    Returns:
        TrueColPaliIndexer instance, or None if not yet created.
    """
    return _visual_indexer_instance
