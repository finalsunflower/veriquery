"""
设备对比节点 - 多设备参数对比的核心流水线实现

架构层级: Agent层 (agents/) - 工作流编排与节点实现

调用链路:
    API入口 → 本节点(对比节点) → 参数提取器/混合检索器/知识图谱/评分引擎 → 返回对比结果

核心职责:
    1. 文档检索 - 仅在用户指定文档中搜索参数，避免跨文档数据泄露
    2. 参数提取 - SmartParameterExtractor 三阶段流水线(表格→正则→LLM)
    3. 知识图谱补充 - 填充缺失参数(兜底数据源)
    4. 多维评分 - CCM条件标准化 + Z-A-FoM可信度融合 + B-SPOTIS稳健决策
    5. 结果生成 - 对比矩阵、评分排名、雷达图数据、Markdown摘要

设计要点:
    - 延迟初始化: 评分引擎/提取器/知识图谱引擎按需创建
    - LRU缓存: 相同芯片组合评分结果缓存5分钟
    - 三阶段参数提取: 表格查询(O(1)) → 正则匹配 → LLM验证，逐级降级
    - 异步并发: 多芯片参数提取使用 asyncio.gather 并行执行
    - 优雅降级: 知识图谱/智能提取器不可用时自动跳过
"""

import logging
import re
import time
import asyncio
import traceback
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict

from core import AgentState, get_settings
from reasoning.parameter_scorer import EnhancedParameterScoringEngine

logger = logging.getLogger(__name__)

try:
    from knowledge import SQLiteGraphQueryEngine
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KNOWLEDGE_GRAPH_AVAILABLE = False
    logger.warning("知识图谱模块不可用")

try:
    from extraction.parameter_extractor import SmartParameterExtractor
    SMART_EXTRACTOR_AVAILABLE = True
except ImportError:
    SMART_EXTRACTOR_AVAILABLE = False
    logger.warning("智能参数提取器不可用")

COMPARISON_PARAMS = [
    "VCC", "VDD", "VOH", "VOL", "VIH", "VIL",
    "IOH", "IOL", "IIH", "IIL", "ICC", "IDD",
    "tPLH", "tPHL", "tpd", "fmax", "Frequency",
    "Temperature",
]


def _parse_condition_to_dict(condition_str: str) -> Dict[str, float]:
    """将条件字符串解析为结构化测试条件字典。

    示例: "IOH = -4 mA, VCC = 5V, 25C" → {"temperature": 25.0, "voltage": 5.0}
    """
    if not condition_str:
        return {}
    conditions: Dict[str, float] = {}
    import re as _re
    temp_match = _re.search(r'(\d+)\s*C', condition_str, _re.IGNORECASE)
    if temp_match:
        conditions["temperature"] = float(temp_match.group(1))
    vcc_match = _re.search(r'VCC\s*=\s*([\d.]+)\s*V?', condition_str, _re.IGNORECASE)
    if vcc_match:
        conditions["voltage"] = float(vcc_match.group(1))
    vdd_match = _re.search(r'VDD\s*=\s*([\d.]+)\s*V?', condition_str, _re.IGNORECASE)
    if vdd_match and "voltage" not in conditions:
        conditions["voltage"] = float(vdd_match.group(1))
    return conditions


MIN_COMPARISON_CHIPS = 2
STAGE2_MIN_PARAMS_THRESHOLD = 3

PARAM_DISPLAY_NAMES = {
    "VCC": "供电电压", "VDD": "供电电压",
    "VOH": "输出高电平", "VOL": "输出低电平",
    "VIH": "输入高电平", "VIL": "输入低电平",
    "IOH": "输出高电流", "IOL": "输出低电流",
    "IIH": "输入高电流", "IIL": "输入低电流",
    "ICC": "静态电流", "IDD": "静态电流",
    "tPLH": "上升延迟", "tPHL": "下降延迟",
    "tpd": "传播延迟",
    "fmax": "最大频率", "Frequency": "工作频率",
    "Temperature": "工作温度",
}

PARAM_NAME_TO_SCORING = {
    "VCC": "supply_voltage", "VDD": "supply_voltage",
    "VOH": "output_voltage_high", "VOL": "output_voltage_low",
    "VIH": "input_voltage_high", "VIL": "input_voltage_low",
    "IOH": "output_current", "IOL": "output_current",
    "IIH": "input_current", "IIL": "input_current",
    "ICC": "quiescent_current", "IDD": "quiescent_current",
    "tPLH": "propagation_delay", "tPHL": "propagation_delay",
    "tpd": "propagation_delay",
    "fmax": "frequency", "Frequency": "frequency",
    "Temperature": "temperature_range",
}

KG_PARAM_MAP = {
    "VOH": ["VOH", "output_voltage_high"],
    "VOL": ["VOL", "output_voltage_low"],
    "VIH": ["VIH", "input_voltage_high"],
    "VIL": ["VIL", "input_voltage_low"],
    "IOH": ["IOH", "output_current"],
    "IOL": ["IOL", "output_current"],
    "IIH": ["IIH", "input_current"],
    "IIL": ["IIL", "input_current"],
    "ICC": ["ICC", "quiescent_current"],
    "IDD": ["IDD", "quiescent_current"],
    "VCC": ["VCC", "supply_voltage"],
    "VDD": ["VDD", "supply_voltage"],
    "tPLH": ["tPLH", "propagation_delay"],
    "tPHL": ["tPHL", "propagation_delay"],
    "tpd": ["tpd", "propagation_delay"],
    "fmax": ["fmax", "frequency", "Frequency"],
    "Frequency": ["Frequency", "fmax", "frequency"],
    "Temperature": ["Temperature", "temperature_range"],
}

CHIP_NAME_PATTERN = re.compile(
    r'\b([A-Z]{2,}\d+[A-Z0-9-]*|SN\d+[A-Z]+\d*|NE\d+|LM\d+|CD\d+[A-Z]*)\b',
    re.IGNORECASE,
)

DEFAULT_TEMPERATURE_RANGE = 85.0
DEFAULT_PACKAGE_SIZE_MM2 = 14.0
DEFAULT_VALUE_CONFIDENCE = 0.3


class ComparisonNodeEnhanced:
    """设备参数对比节点，执行多设备参数对比的完整流水线。

    支持三阶段递进式参数提取、知识图谱兜底补充、三层评分引擎(CCM+Z-A-FoM+B-SPOTIS)，
    并生成对比矩阵、评分排名、雷达图数据和Markdown摘要。
    """

    _CACHE_MAX_SIZE = 50
    _CACHE_TTL = 300.0

    def __init__(
        self,
        retriever=None,
        settings=None,
        llm_client=None,
        table_store=None,
        enable_knowledge_graph: bool = True,
    ):
        self.settings = settings or get_settings()
        self.retriever = retriever
        self._scoring_engine = None
        self.llm_client = llm_client
        self.table_store = table_store
        self.enable_knowledge_graph = enable_knowledge_graph and KNOWLEDGE_GRAPH_AVAILABLE
        self._smart_extractor = None
        self._kg_engine = None
        self._scoring_cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        logger.debug("对比节点初始化完成")

    @property
    def scoring_engine(self) -> EnhancedParameterScoringEngine:
        """延迟初始化评分引擎。"""
        if self._scoring_engine is None:
            self._scoring_engine = EnhancedParameterScoringEngine()
        return self._scoring_engine

    def _get_smart_extractor(self) -> Optional[SmartParameterExtractor]:
        """延迟初始化智能参数提取器。"""
        if self._smart_extractor is None and SMART_EXTRACTOR_AVAILABLE:
            self._smart_extractor = SmartParameterExtractor(
                llm_client=self.llm_client,
                table_store=self.table_store,
            )
        return self._smart_extractor

    def _get_kg_engine(self) -> Optional[SQLiteGraphQueryEngine]:
        """延迟初始化知识图谱引擎。"""
        if self._kg_engine is None and self.enable_knowledge_graph:
            self._kg_engine = SQLiteGraphQueryEngine()
        return self._kg_engine

    def _get_cached_scoring(self, devices_data: Dict[str, Dict[str, Any]]) -> Any:
        """评分缓存，LRU策略避免对相同芯片组合重复计算。"""
        cache_key = str(tuple(sorted(devices_data.keys())))
        current_time = time.time()

        if cache_key in self._scoring_cache:
            cached_result, cached_time = self._scoring_cache[cache_key]
            if current_time - cached_time < self._CACHE_TTL:
                self._scoring_cache.move_to_end(cache_key)
                logger.debug(f"评分缓存命中: {list(devices_data.keys())}")
                return cached_result
            else:
                del self._scoring_cache[cache_key]

        logger.debug(f"评分计算: {list(devices_data.keys())}")
        result = self.scoring_engine.score_devices(devices_data=devices_data)
        self._scoring_cache[cache_key] = (result, current_time)

        if len(self._scoring_cache) > self._CACHE_MAX_SIZE:
            self._scoring_cache.popitem(last=False)

        return result

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行对比分析，本节点的唯一入口方法。

        Args:
            state: LangGraph状态字典，包含:
                - query: 用户原始查询文本
                - chips: 芯片型号列表
                - selected_document_ids: 用户选定的文档ID列表

        Returns:
            更新后的state字典，新增:
                - comparison_matrix: 对比矩阵
                - scoring_result: 评分结果
                - radar_data: 雷达图数据
                - summary: Markdown格式对比摘要
                - processing_time: 处理耗时(秒)
                - intent: 固定为"comparison"
        """
        start_time = time.time()

        try:
            query = state.get("query", "")
            chips = state.get("chips", [])
            document_ids = state.get("selected_document_ids", [])

            if not chips:
                matches = CHIP_NAME_PATTERN.findall(query)
                chips = list(set(c.upper() for c in matches))[:5]

            if len(chips) < MIN_COMPARISON_CHIPS:
                return {
                    **state,
                    "final_response": "请提供至少两个芯片型号进行对比分析。",
                    "comparison_matrix": None,
                    "intent": "comparison",
                }

            logger.debug(f"对比分析开始: {chips}")

            tasks = [self._get_chip_parameters_complete(chip, document_ids) for chip in chips]
            results = await asyncio.gather(*tasks)

            comparison_data = {}
            all_params = set()

            for chip, result in zip(chips, results):
                comparison_data[chip] = result["params"]
                all_params.update(result["params"].keys())

            matrix_params = [
                p for p in COMPARISON_PARAMS if p in all_params
            ] + [
                p for p in all_params if p not in COMPARISON_PARAMS and p in PARAM_DISPLAY_NAMES
            ]

            comparison_matrix = {
                "chips": chips,
                "parameters": matrix_params if matrix_params else list(PARAM_DISPLAY_NAMES.keys()),
                "data": comparison_data,
            }

            scoring_data = self._convert_params_for_scoring(comparison_data)
            scoring_result = self._get_cached_scoring(scoring_data)
            scoring_dict = self._scoring_result_to_dict(scoring_result)
            radar_data = self._generate_radar_data(scoring_result)
            summary = self._generate_comparison_summary(comparison_matrix, scoring_result)

            return {
                **state,
                "comparison_matrix": comparison_matrix,
                "scoring_result": scoring_dict,
                "radar_data": radar_data,
                "summary": summary,
                "processing_time": time.time() - start_time,
                "intent": "comparison",
            }

        except ValueError as e:
            logger.error(f"参数验证失败: {e}")
            logger.error(traceback.format_exc())
            return {
                **state,
                "error": f"参数验证失败: {str(e)}",
                "final_response": f"参数验证失败: {str(e)}",
            }
        except RuntimeError as e:
            logger.error(f"运行时错误: {e}")
            logger.error(traceback.format_exc())
            return {
                **state,
                "error": f"运行时错误: {str(e)}",
                "final_response": f"运行时错误: {str(e)}",
            }
        except Exception as e:
            logger.error(f"对比分析失败: {e}")
            logger.error(traceback.format_exc())
            return {
                **state,
                "error": str(e),
                "final_response": f"对比分析失败: {str(e)}",
            }

    def _extract_numeric_value(self, param_value: Any) -> Optional[float]:
        """从参数值中提取数值，兼容dict格式和原始值格式。"""
        if isinstance(param_value, dict):
            value = param_value.get("value")
        else:
            value = param_value

        if value is not None:
            try:
                return float(value)
            except (ValueError, TypeError):
                pass
        return None

    def _convert_params_for_scoring(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """将参数名转换为评分引擎格式，并填充推断参数和默认值。

        转换内容:
            1. 参数名映射: VCC→supply_voltage, VOH→output_voltage_high 等
            2. 功耗推断: P = |ICC| × VCC / 1000 (mW)
            3. 默认值填充: temperature_range=85℃, package_size=14mm²
            4. TTL兼容性推断: 芯片名含"HCT"/"TTL"/"LVTTL"则兼容
        """
        scoring_data = {}

        for chip_name, params in comparison_data.items():
            scoring_data[chip_name] = {}

            for param_name, param_value in params.items():
                scoring_name = PARAM_NAME_TO_SCORING.get(param_name, param_name.lower())
                value = self._extract_numeric_value(param_value)

                if value is not None:
                    source = param_value.get("source", "datasheet_table") if isinstance(param_value, dict) else "datasheet_table"
                    confidence = param_value.get("confidence", 0.8) if isinstance(param_value, dict) else 0.8
                    test_conditions = param_value.get("test_conditions", {}) if isinstance(param_value, dict) else {}
                    scoring_data[chip_name][scoring_name] = {
                        "value": value,
                        "source": source,
                        "confidence": confidence,
                        "test_conditions": test_conditions,
                    }

            if "quiescent_current" in scoring_data[chip_name] and "power_consumption" not in scoring_data[chip_name]:
                q_current = scoring_data[chip_name]["quiescent_current"]
                if isinstance(q_current, dict) and q_current.get("value"):
                    supply_v = scoring_data[chip_name].get("supply_voltage", {})
                    supply_val = supply_v.get("value", 5.0) if isinstance(supply_v, dict) else 5.0
                    try:
                        power = abs(float(q_current["value"])) * float(supply_val) / 1000
                        scoring_data[chip_name]["power_consumption"] = {
                            "value": power,
                            "source": "llm_inference",
                            "confidence": 0.7,
                        }
                    except (ValueError, TypeError):
                        pass

            if "temperature_range" not in scoring_data[chip_name]:
                scoring_data[chip_name]["temperature_range"] = {
                    "value": DEFAULT_TEMPERATURE_RANGE,
                    "source": "default_value",
                    "confidence": DEFAULT_VALUE_CONFIDENCE,
                }

            if "package_size" not in scoring_data[chip_name]:
                scoring_data[chip_name]["package_size"] = {
                    "value": DEFAULT_PACKAGE_SIZE_MM2,
                    "source": "default_value",
                    "confidence": DEFAULT_VALUE_CONFIDENCE,
                }

            chip_name_upper = chip_name.upper()
            if "ttl_compatible" not in scoring_data[chip_name]:
                is_ttl = "HCT" in chip_name_upper or "TTL" in chip_name_upper or "LVTTL" in chip_name_upper
                scoring_data[chip_name]["ttl_compatible"] = {
                    "value": 1.0 if is_ttl else 0.0,
                    "source": "rule_inference",
                    "confidence": 0.7,
                }

        return scoring_data

    async def _get_chip_parameters_complete(
        self,
        chip: str,
        document_ids: List[str],
    ) -> Dict[str, Any]:
        """完整的参数提取流水线: 三阶段递进式提取 + 知识图谱兜底。

        阶段1: SmartParameterExtractor 智能提取(表格→正则→LLM)，仅使用指定文档
        阶段2: 检索回退，仅当阶段1提取参数 < 3个时触发
        阶段3: 知识图谱补充，仅补充阶段1和2都未提取到的参数，含物理合理性校验
        """
        params: Dict[str, Any] = {}

        # 阶段1: SmartParameterExtractor 智能提取
        smart_extractor = self._get_smart_extractor()
        if smart_extractor and document_ids:
            try:
                logger.debug(f"[阶段1] 智能参数提取器提取: {chip}")
                specs = await smart_extractor.extract_batch(
                    text="",
                    target_params=list(COMPARISON_PARAMS),
                    chip_name=chip,
                    document_ids=document_ids,
                )
                for spec in specs:
                    param = spec.param.upper()
                    if param in COMPARISON_PARAMS or param in PARAM_DISPLAY_NAMES:
                        value = spec.typ_value or spec.max_value or spec.min_value
                        cond_str = spec.condition if hasattr(spec, 'condition') and spec.condition else ""
                        params[param] = {
                            "value": float(value) if value is not None else None,
                            "display": f"{value} {spec.unit}" if value is not None else "N/A",
                            "unit": spec.unit,
                            "source": spec.source_type.value,
                            "confidence": spec.confidence,
                            "condition": cond_str,
                            "test_conditions": _parse_condition_to_dict(cond_str),
                        }
                logger.debug(f"[阶段1] 提取完成: {len(params)} 参数")
            except Exception as e:
                logger.warning(f"智能参数提取器失败: {e}")

        # 阶段2: 检索回退(仅当阶段1提取参数不足时触发)
        if self.retriever and document_ids and len(params) < STAGE2_MIN_PARAMS_THRESHOLD:
            try:
                logger.debug(f"[阶段2] 检索补充: {chip}")
                results = await self.retriever.retrieve(
                    f"{chip} {' '.join(COMPARISON_PARAMS[:5])}",
                    document_ids=document_ids,
                )
                if results:
                    combined_text = ""
                    for result in results[:2]:
                        text = result.text if hasattr(result, 'text') else result.get("content", "")
                        combined_text += text[:1500] + "\n\n"

                    if combined_text.strip() and smart_extractor:
                        missing_params = [p for p in COMPARISON_PARAMS if p not in params]
                        if missing_params:
                            specs = await smart_extractor.extract_batch(
                                text=combined_text,
                                target_params=missing_params,
                                chip_name=chip,
                                document_ids=None,
                            )
                            for spec in specs:
                                param = spec.param.upper()
                                if param not in params and param in COMPARISON_PARAMS:
                                    value = spec.typ_value or spec.max_value or spec.min_value
                                    cond_str = spec.condition if hasattr(spec, 'condition') and spec.condition else ""
                                    params[param] = {
                                        "value": float(value) if value is not None else None,
                                        "display": f"{value} {spec.unit}" if value is not None else "N/A",
                                        "unit": spec.unit,
                                        "source": "retrieval",
                                        "confidence": spec.confidence,
                                        "condition": cond_str,
                                        "test_conditions": _parse_condition_to_dict(cond_str),
                                    }
                            logger.debug(f"[阶段2] 检索补充完成: {len(params)} 参数")
            except Exception as e:
                logger.warning(f"检索失败: {e}")

        # 阶段3: 知识图谱补充(仅补充阶段1和2都未提取到的参数)
        missing_params = [p for p in COMPARISON_PARAMS if p not in params]
        if missing_params and self.enable_knowledge_graph:
            try:
                kg_engine = self._get_kg_engine()
                if kg_engine:
                    logger.debug(f"[阶段3] 知识图谱补充: {chip}")
                    kg_params = kg_engine.query_chip_parameters(chip)

                    for param in missing_params:
                        kg_keys = KG_PARAM_MAP.get(param, [param])
                        for kg_key in kg_keys:
                            if kg_key in kg_params:
                                kg_param = kg_params[kg_key]
                                raw_value = kg_param.get("value")
                                if raw_value is None:
                                    continue
                                try:
                                    value = float(raw_value)
                                except (ValueError, TypeError):
                                    continue

                                # 物理合理性校验
                                if param in ['VOH', 'VIH'] and value < 0.5:
                                    continue
                                if param in ['VOL', 'VIL'] and value > 5.0:
                                    continue
                                # 电流符号修正: IOH为拉电流取负值，IOL为灌电流取正值
                                if param == 'IOH' and value > 0:
                                    value = -abs(value)
                                if param == 'IOL' and value < 0:
                                    value = abs(value)

                                params[param] = {
                                    "value": value,
                                    "display": f"{value} {kg_param.get('unit', '')}",
                                    "unit": kg_param.get("unit", ""),
                                    "source": kg_param.get("source", "knowledge_graph"),
                                    "confidence": kg_param.get("confidence", 0.7),
                                    "condition": kg_param.get("condition", ""),
                                    "test_conditions": _parse_condition_to_dict(kg_param.get("condition", "")),
                                }
                                break

                    logger.debug(f"[阶段3] 知识图谱补充完成: {len(params)} 参数")
            except Exception as e:
                logger.warning(f"知识图谱查询失败: {e}")

        return {"params": params}

    def _scoring_result_to_dict(self, result: Any) -> Dict[str, Any]:
        """将评分结果对象转换为可序列化的字典格式。"""
        if not result:
            return {}

        try:
            devices = []
            for d in result.devices:
                device_dict = {
                    "device_name": d.device_name,
                    "overall_score": d.overall_score,
                    "reliability_score": d.reliability_score,
                    "dimension_scores": d.dimension_scores,
                    "parameter_scores": d.parameter_scores,
                    "parameter_reliabilities": d.parameter_reliabilities,
                    "advantages": d.advantages,
                    "disadvantages": d.disadvantages,
                    "esp_distance": d.esp_distance,
                    "rank": d.rank,
                }
                devices.append(device_dict)

            objective_weights = {k: float(v) for k, v in result.objective_weights.items()} if result.objective_weights else {}
            reliability_weights = {k: float(v) for k, v in result.reliability_weights.items()} if result.reliability_weights else {}
            parameter_weights = {k: float(v) for k, v in result.parameter_weights.items()} if result.parameter_weights else {}

            return {
                "devices": devices,
                "recommendation": result.recommendation,
                "methodology": result.methodology,
                "combined_weights": objective_weights,
                "entropy_weights": reliability_weights,
                "parameter_weights": parameter_weights,
            }
        except Exception as e:
            logger.warning(f"评分结果转换失败: {e}")
            return {}

    def _generate_comparison_summary(self, matrix: Dict[str, Any], result: Any) -> str:
        """生成Markdown格式的对比摘要。"""
        chips = matrix.get("chips", [])
        data = matrix.get("data", {})

        lines = []
        lines.append("## 设备对比分析\n")
        lines.append(f"**对比设备**: {', '.join(chips)}\n\n")
        lines.append("### 参数对比\n")

        key_params = ["VCC", "VOH", "VOL", "VIH", "VIL", "IOH", "IOL", "tpd"]
        lines.append("| 参数 | " + " | ".join(chips) + " |\n")
        lines.append("|" + "---|" * (len(chips) + 1) + "\n")

        for param in key_params:
            display_name = PARAM_DISPLAY_NAMES.get(param, param)
            row = [f"**{display_name}**"]
            for chip in chips:
                cell = data.get(chip, {}).get(param, {})
                if isinstance(cell, dict):
                    display = cell.get("display", "N/A")
                else:
                    display = str(cell) if cell else "N/A"
                row.append(display if display and display != "N/A" else "-")
            lines.append("| " + " | ".join(row) + " |\n")

        if result:
            lines.append(f"\n### 分析结果\n")
            if result.recommendation:
                lines.append(f"{result.recommendation}\n")
            if result.devices:
                lines.append("\n**评分排名**:\n")
                sorted_devices = sorted(result.devices, key=lambda d: d.overall_score, reverse=True)
                for device in sorted_devices:
                    lines.append(f"- {device.device_name}: {device.overall_score:.2f}\n")

        return "".join(lines)

    def _generate_radar_data(self, comparison_result: Any) -> Dict[str, Any]:
        """生成雷达图数据，供前端ECharts渲染。

        四个评分维度: performance(性能), power(功耗), reliability(可靠性), usability(易用性)
        """
        if not comparison_result or not comparison_result.devices:
            return {}

        devices_data = comparison_result.devices
        if not devices_data:
            return {}

        dim_map = {
            "performance": "性能",
            "power": "功耗",
            "reliability": "可靠性",
            "usability": "易用性",
        }

        devices = []
        for device in devices_data:
            device_dict = {
                "name": device.device_name,
                "values": [],
            }
            dimension_scores = device.dimension_scores if device.dimension_scores else {}
            for dim_key in dim_map.keys():
                score = dimension_scores.get(dim_key, 0.0) if isinstance(dimension_scores, dict) else 0.0
                device_dict["values"].append(float(score))
            devices.append(device_dict)

        return {
            "dimensions": list(dim_map.values()),
            "devices": devices,
        }
