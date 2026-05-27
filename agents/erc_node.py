"""
ERC检查节点 - 驱动端/接收端电气兼容性验证的核心编排器

核心职责:
    本节点采用"编排器(Orchestrator)"设计模式，协调参数提取、知识图谱、ERC推理等子模块，
    完成驱动端-接收端芯片电气兼容性验证的完整数据流。

数据流:
    输入state(含driver_chip, receiver_chip, document_ids)
    → Stage 0: HybridRetriever文档检索
    → Stage 1-2: SmartParameterExtractor三阶段流水线(表格→正则→LLM)
    → Stage 3: SQLiteGraphQueryEngine知识图谱补全
    → Stage 4: 参数验证(完整度/置信度)
    → Stage 5: FourLayerERCEngine四层兼容性检测
        (Layer1:静态稳定性 → Layer2:准物理反射 → Layer3:拓扑冲突 → Layer4:极端环境退化)
    → 输出: EnhancedERCResult → Markdown报告

设计要点:
    - 编排器模式: 本节点只协调数据流，检测逻辑委托给FourLayerERCEngine
    - 多源融合: 文档提取+知识图谱交叉验证，提高参数准确性
    - 参数可疑检测: 文档值与KG基准值比对，过滤异常值
    - 缓存机制: 同一芯片参数缓存1小时(TTL=3600s)，避免重复提取
    - 懒加载: SmartParameterExtractor和KG引擎按需初始化
    - 优雅降级: 知识图谱/提取器不可用时自动跳过
"""

import logging
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field

from reasoning import (
    FourLayerERCEngine,
    FourLayerERCResult,
)

logger = logging.getLogger(__name__)

try:
    from knowledge import SQLiteGraphQueryEngine
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KNOWLEDGE_GRAPH_AVAILABLE = False
    logger.warning("Knowledge graph module not available")

try:
    from extraction.parameter_extractor import SmartParameterExtractor
    SMART_EXTRACTOR_AVAILABLE = True
except ImportError:
    SMART_EXTRACTOR_AVAILABLE = False
    logger.warning("SmartParameterExtractor not available")


@dataclass
class CacheEntry:
    """带TTL的缓存条目，用于避免重复提取同一芯片参数。"""

    value: Any
    timestamp: float

    def is_valid(self, ttl_seconds: float) -> bool:
        return (time.time() - self.timestamp) < ttl_seconds


@dataclass
class ParameterValidationResult:
    """参数验证结果，评估提取到的电气参数的完整度和可靠性。"""

    is_complete: bool
    completeness_ratio: float
    missing_params: List[str]
    available_params: List[str]
    data_sources: Dict[str, str]
    confidence: float


@dataclass
class EnhancedERCResult:
    """增强版ERC检查结果，包含检测结论、参数验证信息和元数据。"""

    is_compatible: bool
    driver_chip: str
    receiver_chip: str
    driver_params: Dict[str, Any]
    receiver_params: Dict[str, Any]
    driver_validation: ParameterValidationResult
    receiver_validation: ParameterValidationResult
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


class ERCNodeEnhanced:
    """ERC检查节点，驱动端-接收端电气兼容性验证的完整数据流编排器。

    协调文档检索、参数提取、知识图谱补全、参数验证和四层ERC检测，
    生成增强版检测报告。
    """

    DRIVER_PARAMS = ["VOH", "VOL", "IOH", "IOL"]
    RECEIVER_PARAMS = ["VIH", "VIL", "IIH", "IIL"]

    def __init__(
        self,
        retriever=None,
        erc_engine: Optional[FourLayerERCEngine] = None,
        llm_client=None,
        settings=None,
        enable_knowledge_graph: bool = True,
        table_store=None,
    ):
        self.settings = settings
        self.enable_knowledge_graph = enable_knowledge_graph and KNOWLEDGE_GRAPH_AVAILABLE
        self.retriever = retriever
        self.erc_engine = erc_engine or FourLayerERCEngine()
        self.llm_client = llm_client
        self.table_store = table_store
        self._smart_extractor = None
        self._kg_engine = None
        self._param_cache: Dict[str, CacheEntry] = {}
        self._cache_ttl = 3600
        logger.info("ERC Node (Enhanced) initialized with complete data flow")

    def _flatten_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """将嵌套参数字典转换为扁平格式，供FourLayerERCEngine使用。

        嵌套格式: {"VOH": {"value": 3.3, "unit": "V"}}
        扁平格式: {"VOH": 3.3, "VOH_unit": "V"}
        """
        flat = {}
        for key, val in params.items():
            if isinstance(val, dict):
                v = val.get("value")
                unit = val.get("unit", "")
                if v is not None:
                    try:
                        flat[key] = float(v)
                        if unit:
                            flat[f"{key}_unit"] = unit
                    except (TypeError, ValueError):
                        pass
            elif isinstance(val, (int, float)):
                flat[key] = float(val)
        return flat

    def _get_smart_extractor(self):
        """懒加载初始化SmartParameterExtractor。"""
        if self._smart_extractor is None and SMART_EXTRACTOR_AVAILABLE:
            self._smart_extractor = SmartParameterExtractor(
                llm_client=self.llm_client,
                table_store=self.table_store,
            )
        return self._smart_extractor

    def _get_kg_engine(self):
        """懒加载初始化知识图谱查询引擎。"""
        if self._kg_engine is None and self.enable_knowledge_graph:
            self._kg_engine = SQLiteGraphQueryEngine()
        return self._kg_engine

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行ERC检查的完整数据流，本节点的主入口方法。

        Args:
            state: LangGraph状态字典，包含:
                - query: 用户原始查询文本
                - driver_chip: 驱动端芯片名称
                - receiver_chip: 接收端芯片名称
                - metadata.selected_document_ids: 限定检索的文档ID列表

        Returns:
            更新后的state字典，新增:
                - response: Markdown格式ERC报告
                - erc_result: 四层检测结果字典
                - driver_parameters/receiver_parameters: 参数字典
                - driver_validation/receiver_validation: 参数验证结果
                - overall_compatible: 总体兼容性结论
                - overall_confidence: 总体置信度
                - suggestions: 改进建议列表
        """
        start_time = time.time()

        try:
            query = state.get("query", "")
            metadata = state.get("metadata", {})
            logger.debug(f"[NODE] metadata类型={type(metadata)}, 值={metadata}")
            document_ids = metadata.get("selected_document_ids", []) if isinstance(metadata, dict) else []
            logger.debug(f"[NODE] 最终document_ids={document_ids}")

            driver_chip = state.get("driver_chip")
            receiver_chip = state.get("receiver_chip")

            if not driver_chip or not receiver_chip:
                driver_chip, receiver_chip = self._extract_chips_from_query(query)

            if not driver_chip or not receiver_chip:
                return {
                    **state,
                    "response": "请指定驱动端和接收端芯片进行ERC检查。",
                    "intent": "erc_check",
                    "erc_result": None,
                }

            logger.info(f"ERC检查开始: {driver_chip} -> {receiver_chip}")

            driver_result = await self._get_chip_parameters_complete(
                driver_chip, document_ids, is_driver=True
            )
            receiver_result = await self._get_chip_parameters_complete(
                receiver_chip, document_ids, is_driver=False
            )

            driver_params = driver_result["params"]
            receiver_params = receiver_result["params"]
            driver_validation = driver_result["validation"]
            receiver_validation = receiver_result["validation"]

            flat_driver_params = self._flatten_params(driver_params)
            flat_receiver_params = self._flatten_params(receiver_params)

            topology_info = self._build_topology_info(
                driver_params, receiver_params, driver_chip, receiver_chip
            )
            temperature = self._extract_temperature(state) or 25.0
            trace_length, trace_impedance = self._auto_estimate_trace_params(flat_driver_params)

            erc_result = self.erc_engine.check_four_layer(
                driver_chip=driver_chip,
                driver_params=flat_driver_params,
                receiver_chip=receiver_chip,
                receiver_params=flat_receiver_params,
                topology_info=topology_info,
                temperature=temperature,
                trace_length=trace_length,
                trace_impedance=trace_impedance,
            )

            four_layer_dict = self._convert_four_layer_to_dict(erc_result) if erc_result else {}

            is_compatible = erc_result.overall_compatible if erc_result else False
            overall_confidence = erc_result.overall_confidence if erc_result else 0.0
            suggestions = erc_result.suggestions if erc_result else []
            elapsed_ms = (time.time() - start_time) * 1000

            enhanced_result = EnhancedERCResult(
                is_compatible=is_compatible,
                driver_chip=driver_chip,
                receiver_chip=receiver_chip,
                driver_params=driver_params,
                receiver_params=receiver_params,
                driver_validation=driver_validation,
                receiver_validation=receiver_validation,
                issues=self._extract_issues(erc_result),
                recommendations=suggestions,
                warnings=self._generate_warnings(driver_validation, receiver_validation),
                processing_time_ms=elapsed_ms,
            )
            report = self._generate_enhanced_report(enhanced_result)

            return {
                **state,
                "response": report,
                "erc_result": four_layer_dict,
                "driver_parameters": driver_params,
                "receiver_parameters": receiver_params,
                "driver_data_sources": driver_result.get("data_sources", {}),
                "receiver_data_sources": receiver_result.get("data_sources", {}),
                "driver_validation": asdict(driver_validation),
                "receiver_validation": asdict(receiver_validation),
                "intent": "erc_check",
                "processing_time": elapsed_ms / 1000,
                "overall_compatible": is_compatible,
                "overall_confidence": overall_confidence,
                "suggestions": suggestions,
            }

        except Exception as e:
            logger.error(f"ERC检查失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                **state,
                "error": str(e),
                "response": f"ERC检查失败: {str(e)}",
                "erc_result": None,
            }

    async def _get_chip_parameters_complete(
        self,
        chip: str,
        document_ids: List[str],
        is_driver: bool = True,
    ) -> Dict[str, Any]:
        """完整的参数提取流水线: 四阶段递进式提取。

        Stage 0: 文档检索 - HybridRetriever检索相关文档片段
        Stage 1-2: 参数提取 - SmartParameterExtractor三阶段流水线(表格→正则→LLM)
        Stage 3: 知识图谱补全 - 补充缺失/可疑参数
        Stage 4: 数据验证 - 检查参数完整度，生成验证结果
        """
        cache_key = f"{chip}_{'driver' if is_driver else 'receiver'}"
        if cache_key in self._param_cache:
            entry = self._param_cache[cache_key]
            if entry.is_valid(self._cache_ttl):
                logger.debug(f"缓存命中: {chip}")
                return entry.value

        target_params = self.DRIVER_PARAMS if is_driver else self.RECEIVER_PARAMS
        params: Dict[str, Any] = {}
        data_sources: Dict[str, str] = {}

        # Stage 0: 文档检索
        doc_text = ""
        logger.info(f"[Stage 0] 开始文档检索: chip={chip}, document_ids={document_ids}, retriever={self.retriever is not None}")
        if self.retriever and document_ids:
            try:
                logger.info(f"[Stage 0] 检索文档内容: {chip}")
                results = await self.retriever.retrieve(
                    f"{chip} {' '.join(target_params)} electrical characteristics",
                    document_ids=document_ids,
                    top_k=5,
                )
                logger.info(f"[Stage 0] 检索返回 {len(results) if results else 0} 个结果")
                if results:
                    text_parts = []
                    for r in results[:3]:
                        text = r.text if hasattr(r, 'text') else r.get("content", "")
                        if text:
                            text_parts.append(text)
                    doc_text = "\n\n".join(text_parts)
                    logger.info(f"[Stage 0] 检索到 {len(doc_text)} 字符文档内容")
                else:
                    logger.warning(f"[Stage 0] 检索结果为空")
            except Exception as e:
                logger.warning(f"检索失败: {e}")
        elif not document_ids:
            logger.warning(f"[Stage 0] document_ids 为空，跳过文档检索")
        else:
            logger.warning(f"[Stage 0] retriever 未初始化")

        # Stage 1-2: SmartParameterExtractor参数提取
        smart_extractor = self._get_smart_extractor()
        logger.info(f"[Stage 1-2] smart_extractor={'可用' if smart_extractor else 'None'}, document_ids={document_ids}, doc_text长度={len(doc_text)}")
        doc_specs = {}
        if smart_extractor and document_ids:
            try:
                logger.info(f"[Stage 1-2] SmartParameterExtractor提取: {chip}")
                specs = await smart_extractor.extract_batch(
                    text=doc_text,
                    target_params=target_params,
                    chip_name=chip,
                    document_ids=document_ids,
                )
                for spec in specs:
                    if spec.param in target_params:
                        doc_specs[spec.param] = spec
                logger.info(f"[Stage 1-2] 提取完成: {len(doc_specs)}/{len(target_params)} 参数: {list(doc_specs.keys())}")
            except Exception as e:
                logger.warning(f"SmartParameterExtractor失败: {e}")
        elif not smart_extractor:
            logger.warning(f"[Stage 1-2] SmartParameterExtractor 不可用 (SMART_EXTRACTOR_AVAILABLE={SMART_EXTRACTOR_AVAILABLE})")
        else:
            logger.warning(f"[Stage 1-2] document_ids 为空，跳过文档提取")

        # 将提取结果转为统一嵌套格式
        for param, spec in doc_specs.items():
            value = spec.typ_value or spec.max_value or spec.min_value
            logger.debug(f"[PARAM] {param}: value={value}, source_type={spec.source_type.value if spec.source_type else 'None'}, unit={spec.unit}")
            if value is not None:
                params[param] = {
                    "value": value,
                    "unit": spec.unit,
                    "source": spec.source_type.value,
                    "confidence": spec.confidence,
                }
                data_sources[param] = spec.source_type.value

        # Stage 3: 知识图谱补全
        if self.enable_knowledge_graph:
            try:
                kg_engine = self._get_kg_engine()
                if kg_engine:
                    kg_params = kg_engine.query_chip_parameters(chip)
                    for param in target_params:
                        if param in kg_params and kg_params[param].get("value") is not None:
                            kg_value = kg_params[param].get("value")
                            kg_unit = kg_params[param].get("unit", "")
                            kg_confidence = kg_params[param].get("confidence", 0.9)

                            doc_value = params.get(param, {}).get("value") if param in params else None
                            doc_unit = params.get(param, {}).get("unit", "") if param in params else ""

                            should_use_kg = False
                            reason = ""

                            if param not in params:
                                should_use_kg = True
                                reason = "参数缺失"
                            elif doc_value is not None and self._is_value_suspicious(param, doc_value, doc_unit, kg_value, kg_unit):
                                should_use_kg = True
                                reason = f"文档值可疑(doc={doc_value}{doc_unit}, kg={kg_value}{kg_unit})"
                                logger.warning(f"[KG Override] {chip} {param}: {reason}")
                            elif data_sources.get(param) in ("table", "regex", "llm"):
                                should_use_kg = False
                                reason = f"文档已提取({data_sources[param]})"

                            if should_use_kg:
                                logger.info(f"[KG Override] {chip} {param}: {reason}")
                                params[param] = {
                                    "value": kg_value,
                                    "unit": kg_unit,
                                    "source": "knowledge_graph",
                                    "confidence": kg_confidence,
                                }
                                data_sources[param] = "knowledge_graph"
            except Exception as e:
                logger.warning(f"知识图谱查询失败: {e}")

        missing_params = [p for p in target_params if p not in params]
        if missing_params:
            logger.info(f"[Stage 3] 仍有缺失参数: {missing_params}")

        # Stage 4: 数据验证
        validation = self._validate_parameters(params, target_params, data_sources)

        result = {
            "params": params,
            "validation": validation,
            "data_sources": data_sources,
        }

        self._param_cache[cache_key] = CacheEntry(result, time.time())
        return result

    def _validate_parameters(
        self,
        params: Dict[str, Any],
        target_params: List[str],
        data_sources: Dict[str, str],
    ) -> ParameterValidationResult:
        """验证参数完整度，评估提取到的参数是否足够进行可靠的ERC检查。"""
        available = [p for p in target_params if p in params and params[p].get("value") is not None]
        missing = [p for p in target_params if p not in available]
        completeness = len(available) / len(target_params) if target_params else 0.0

        avg_confidence = 0.0
        if params:
            confidences = [p.get("confidence", 0.5) for p in params.values() if isinstance(p, dict)]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        return ParameterValidationResult(
            is_complete=len(missing) == 0,
            completeness_ratio=completeness,
            missing_params=missing,
            available_params=available,
            data_sources=data_sources,
            confidence=avg_confidence,
        )

    def _is_value_suspicious(
        self,
        param: str,
        doc_value: float,
        doc_unit: str,
        kg_value: float,
        kg_unit: str,
    ) -> bool:
        """判断文档提取值是否可疑(与知识图谱基准值偏差过大)。"""
        param_upper = param.upper()

        voltage_suspicious_rules = {
            'VIH': (1.5, 4.0, 0.3),
            'VIL': (0.3, 1.5, 0.3),
            'VOH': (2.0, 5.0, 0.3),
            'VOL': (0.05, 1.5, 0.3),
        }

        if param_upper in voltage_suspicious_rules:
            min_val, max_val, threshold = voltage_suspicious_rules[param_upper]
            if doc_value < min_val or doc_value > max_val:
                return True
            if abs(doc_value - kg_value) > threshold:
                return True
            return False

        if param_upper in ["IIH", "IIL"]:
            doc_ma = self._normalize_current_param_to_ma(doc_value, doc_unit)
            kg_ma = self._normalize_current_param_to_ma(kg_value, kg_unit)
            if abs(doc_ma) > 0.1:
                return True
            if abs(doc_ma) > 0 and abs(kg_ma) > 0:
                ratio = abs(doc_ma) / abs(kg_ma)
                if ratio > 10 or ratio < 0.1:
                    return True
            if abs(doc_ma - kg_ma) > 0.01:
                return True
            return False

        if param_upper in ["IOH", "IOL"]:
            doc_ma = self._normalize_current_param_to_ma(doc_value, doc_unit)
            kg_ma = self._normalize_current_param_to_ma(kg_value, kg_unit)
            if abs(doc_ma) > 1000 or abs(doc_ma) < 0.1:
                return True
            if abs(doc_ma) > 0 and abs(kg_ma) > 0:
                ratio = abs(doc_ma) / abs(kg_ma)
                if ratio > 10 or ratio < 0.1:
                    return True
            return False

        threshold = max(abs(kg_value) * 0.1, 0.0001)
        return abs(doc_value - kg_value) > threshold

    @staticmethod
    def _normalize_current_param_to_ma(value: float, unit: str) -> float:
        """将电流参数归一化为mA单位。"""
        if not unit:
            if abs(value) > 100:
                return value
            elif abs(value) > 0.5:
                return value
            elif abs(value) > 0.001:
                return value * 1000
            else:
                return value * 1000000
        u = unit.lower().strip().replace("μ", "µ")
        if u in ("µa", "ua"):
            return value / 1000.0
        elif u in ("ma",):
            return value
        elif u == "a":
            return value * 1000.0
        return value

    def _convert_four_layer_to_dict(self, erc_result: Any) -> Dict[str, Any]:
        """将FourLayerERCResult转换为dict格式，供前端JSON序列化。"""
        if not erc_result:
            return {}

        if hasattr(erc_result, '__dict__'):
            result_dict = {
                'overall_compatible': getattr(erc_result, 'overall_compatible', False),
                'overall_confidence': getattr(erc_result, 'overall_confidence', 0.0),
                'summary': getattr(erc_result, 'summary', ''),
                'suggestions': getattr(erc_result, 'suggestions', []),
            }

            for layer_name in ['layer1_result', 'layer2_result', 'layer3_result', 'layer4_result']:
                layer = getattr(erc_result, layer_name, None)
                if layer:
                    if hasattr(layer, '__dict__'):
                        layer_dict = {}
                        for key, val in layer.__dict__.items():
                            if key == 'results' and val:
                                layer_dict[key] = [
                                    r.__dict__ if hasattr(r, '__dict__') else r
                                    for r in val
                                ]
                            else:
                                layer_dict[key] = val
                        result_dict[layer_name] = layer_dict
                    else:
                        result_dict[layer_name] = layer

            return result_dict

        return erc_result if isinstance(erc_result, dict) else {}

    def _extract_issues(self, erc_result: Any) -> List[str]:
        """从四层ERC结果中提取所有未通过的规则消息。"""
        issues = []
        if not erc_result:
            return issues

        for layer_name in ['layer1_result', 'layer2_result', 'layer3_result', 'layer4_result']:
            layer = getattr(erc_result, layer_name, None)
            if layer and hasattr(layer, 'results') and layer.results:
                for r in layer.results:
                    if hasattr(r, 'passed') and not r.passed and hasattr(r, 'message'):
                        issues.append(r.message)

        if hasattr(erc_result, 'summary') and erc_result.summary:
            if not issues:
                issues.append(erc_result.summary)

        return issues

    def _generate_warnings(
        self,
        driver_validation: ParameterValidationResult,
        receiver_validation: ParameterValidationResult,
    ) -> List[str]:
        """根据参数验证结果生成警告信息。"""
        warnings = []

        if driver_validation.missing_params:
            warnings.append(f"驱动端缺失参数: {', '.join(driver_validation.missing_params)}")

        if receiver_validation.missing_params:
            warnings.append(f"接收端缺失参数: {', '.join(receiver_validation.missing_params)}")

        if driver_validation.completeness_ratio < 0.5:
            warnings.append("驱动端参数完整度低于50%，检查结果可能不准确")

        if receiver_validation.completeness_ratio < 0.5:
            warnings.append("接收端参数完整度低于50%，检查结果可能不准确")

        return warnings

    def _generate_enhanced_report(self, result: EnhancedERCResult) -> str:
        """生成增强版ERC检查报告(Markdown格式)。"""
        lines = []

        lines.append("## ERC 电气规则兼容性检查报告\n")
        lines.append(f"**驱动端**: {result.driver_chip}\n")
        lines.append(f"**接收端**: {result.receiver_chip}\n")
        lines.append(f"**处理时间**: {result.processing_time_ms:.1f}ms\n\n")

        status_icon = "✅" if result.is_compatible else "❌"
        status_text = "兼容" if result.is_compatible else "不兼容"
        lines.append(f"### 检查结果: {status_icon} {status_text}\n\n")

        lines.append("### 参数完整度\n")
        lines.append(f"| 芯片 | 完整度 | 可用参数 | 缺失参数 |\n")
        lines.append(f"|------|--------|----------|----------|\n")

        dv = result.driver_validation
        rv = result.receiver_validation

        lines.append(f"| {result.driver_chip} | {dv.completeness_ratio*100:.0f}% | {', '.join(dv.available_params) or '无'} | {', '.join(dv.missing_params) or '无'} |\n")
        lines.append(f"| {result.receiver_chip} | {rv.completeness_ratio*100:.0f}% | {', '.join(rv.available_params) or '无'} | {', '.join(rv.missing_params) or '无'} |\n\n")

        if result.warnings:
            lines.append("### ⚠️ 警告\n")
            for warning in result.warnings:
                lines.append(f"- {warning}\n")
            lines.append("\n")

        if result.issues:
            lines.append("### 问题详情\n")
            for issue in result.issues:
                lines.append(f"- {issue}\n")
            lines.append("\n")

        if result.recommendations:
            lines.append("### 建议\n")
            for rec in result.recommendations:
                lines.append(f"- {rec}\n")

        return "".join(lines)

    def _extract_chips_from_query(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """从用户查询文本中正则匹配提取驱动端和接收端芯片名称。

        支持12种常见芯片型号前缀，匹配后去重(大小写不敏感+前缀/后缀包含关系)，
        最多取前2个匹配结果: 第一个为驱动端，第二个为接收端。
        """
        patterns = [
            r'(SN74[A-Z]{0,3}\d+[A-Z0-9-]*)',
            r'(74[A-Z]{1,3}\d+[A-Z0-9-]*)',
            r'(NE\d+[A-Z0-9]*)',
            r'(LM\d+[A-Z0-9]*)',
            r'(CD\d+[A-Z0-9]*)',
            r'(MAX\d+[A-Z0-9]*)',
            r'(MCP\d+[A-Z0-9]*)',
            r'(PCA\d+[A-Z0-9]*)',
            r'(PCF\d+[A-Z0-9]*)',
            r'(ADS\d+[A-Z0-9]*)',
            r'(OP\d+[A-Z0-9]*)',
            r'(TL\d+[A-Z0-9]*)',
        ]

        chips = []
        remaining_text = query
        for pattern in patterns:
            matches = re.findall(pattern, remaining_text, re.IGNORECASE)
            for m in matches:
                upper_m = m.upper()
                is_duplicate = any(
                    upper_m == c.upper() or upper_m.endswith(c.upper()) or c.upper().endswith(upper_m)
                    for c in chips
                )
                if not is_duplicate:
                    chips.append(upper_m)
                    remaining_text = remaining_text.replace(m, '', 1)

        chips = chips[:2]

        if len(chips) >= 2:
            return chips[0], chips[1]
        elif len(chips) == 1:
            return chips[0], None
        return None, None

    def _infer_interface_type(self, chip_name: str, params: Dict[str, Any]) -> str:
        """从芯片名称和参数推断接口类型，供Layer3拓扑冲突仲裁使用。

        推断策略: 优先检查params中显式指定的interface_type，
        其次基于芯片名称关键词推断，默认为GPIO。
        """
        if isinstance(params.get("interface_type"), dict):
            return params["interface_type"].get("value", "GPIO")
        explicit = params.get("interface_type")
        if explicit and isinstance(explicit, str):
            return explicit

        chip_upper = chip_name.upper()

        if any(kw in chip_upper for kw in ["UART", "USART", "SC16IS", "MAX310"]):
            return "UART"
        if any(kw in chip_upper for kw in ["I2C", "IIC", "PCA95", "TCA95", "PCF85"]):
            return "I2C"
        if any(kw in chip_upper for kw in ["SPI", "MCP23S", "74HC595", "74HC165"]):
            return "SPI"
        if any(kw in chip_upper for kw in ["CAN", "MCP25", "SJA100"]):
            return "CAN"
        if any(kw in chip_upper for kw in ["ADC", "MCP30", "ADS", "MAX12"]):
            return "ADC"
        if any(kw in chip_upper for kw in ["DAC", "MCP49", "AD5"]):
            return "DAC"

        return "GPIO"

    def _get_voltage_from_params(self, params: Dict[str, Any], keys: List[str]) -> Optional[float]:
        """从参数字典中提取电压值，支持嵌套格式和简单格式。"""
        for key in keys:
            val = params.get(key)
            if val is not None:
                if isinstance(val, dict):
                    v = val.get("value")
                    if isinstance(v, (int, float)):
                        return float(v)
                elif isinstance(val, (int, float)):
                    return float(val)
        return None

    def _build_topology_info(
        self,
        driver_params: Dict[str, Any],
        receiver_params: Dict[str, Any],
        driver_chip: str = "",
        receiver_chip: str = "",
    ) -> Dict[str, Any]:
        """构建拓扑信息，供Layer3拓扑冲突仲裁使用。

        包含驱动端/接收端的接口类型和供电电压。
        """
        driver_interface = self._infer_interface_type(driver_chip, driver_params)
        receiver_interface = self._infer_interface_type(receiver_chip, receiver_params)

        driver_voltage = self._get_voltage_from_params(driver_params, ["VCC", "supply_voltage", "VDD"])
        receiver_voltage = self._get_voltage_from_params(receiver_params, ["VCC", "supply_voltage", "VDD"])

        return {
            "driver_interface": driver_interface,
            "receiver_interface": receiver_interface,
            "driver_voltage": driver_voltage or 5.0,
            "receiver_voltage": receiver_voltage or 5.0,
        }

    def _extract_temperature(self, state: Dict[str, Any]) -> Optional[float]:
        """从state中提取工作温度，优先显式指定，默认25℃(室温)。"""
        if "temperature" in state and state["temperature"] is not None:
            return float(state["temperature"])

        processing_options = state.get("processing_options", {})
        if isinstance(processing_options, dict) and "temperature" in processing_options:
            temp = processing_options["temperature"]
            if temp is not None:
                return float(temp)

        return None

    def _auto_estimate_trace_params(self, driver_params: Dict[str, float]) -> Tuple[float, float]:
        """自动估算走线参数(走线长度和走线阻抗)，供Layer2准物理反射预警使用。

        基于信号边沿时间估算走线长度: L = t_edge × v_signal / 10
        其中 v_signal ≈ 15cm/ns (FR4介质)，/10为经验系数(集总参数模型适用条件)。
        """
        signal_speed_cm_per_ns = 15.0

        rise_time_s = self._normalize_time_param(driver_params, "rise_time", "tr")
        fall_time_s = self._normalize_time_param(driver_params, "fall_time", "tf")

        if rise_time_s and fall_time_s:
            min_edge_time_ns = min(rise_time_s, fall_time_s) * 1e9
            trace_length_cm = min_edge_time_ns * signal_speed_cm_per_ns / 10
            logger.info(f"[Auto-Estimate] trace_length={trace_length_cm:.2f}cm (based on edge_time={min_edge_time_ns:.2f}ns)")
        else:
            trace_length_cm = 5.0
            logger.info(f"[Auto-Estimate] trace_length={trace_length_cm:.2f}cm (default, no edge_time data)")

        trace_impedance_ohm = 50.0

        return trace_length_cm, trace_impedance_ohm

    def _normalize_time_param(self, params: Dict[str, float], primary_key: str, fallback_key: str) -> Optional[float]:
        """将时间参数归一化为秒(根据单位后缀自动转换)。

        Args:
            params: 扁平格式的参数字典(含"{key}_unit"单位字段)
            primary_key: 首选键名(如"rise_time")
            fallback_key: 备选键名(如"tr")
        """
        value = params.get(primary_key) or params.get(fallback_key)
        if value is None:
            return None

        unit = params.get(f"{primary_key}_unit") or params.get(f"{fallback_key}_unit") or ""

        if unit in ("s", "sec", "seconds"):
            return float(value)
        elif unit in ("ms", "msec", "milliseconds"):
            return float(value) * 1e-3
        elif unit in ("us", "μs", "usec", "microseconds"):
            return float(value) * 1e-6
        elif unit in ("ns", "nsec", "nanoseconds"):
            return float(value) * 1e-9
        else:
            # 无单位时推断: 数字芯片边沿时间通常为ns级
            if abs(value) > 0.001:
                return float(value) * 1e-9
            return float(value)
