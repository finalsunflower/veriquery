"""
ERC (Electrical Rule Compatibility) check router.

Provides API endpoints for checking electrical compatibility between
driver and receiver chips using a four-layer ERC detection engine:

  Layer 1 - Static stability: voltage levels, noise margins, drive capability
  Layer 2 - Quasi-physical reflection: critical transmission line length, reflection coefficient
  Layer 3 - Topology conflict: interface contract semantics, port attribute matrix conflicts
  Layer 4 - Extreme environment degradation: temperature drift, process degradation

Endpoints:
  POST /erc/check  → Run ERC compatibility check between two chips
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core import create_initial_state, MetadataState, AgentState
from agents.erc_node import ERCNodeEnhanced
from api.dependencies import get_service_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/erc", tags=["erc"])

LAYER_KEYS = ["layer1_result", "layer2_result", "layer3_result", "layer4_result"]

DEFAULT_ERROR_RESPONSE = {
    "success": False,
    "message": "ERC检查失败",
    "errors": [{"title": "ERC检查失败", "type": "ERROR", "description": "未知错误"}],
    "warnings": [],
    "info": [],
    "summary": "ERC检查失败",
    "erc_result": None,
    "driver_parameters": {},
    "receiver_parameters": {},
}


class ERCCheckRequest(BaseModel):
    """Request body for ERC compatibility check.

    Attributes:
        driver_chip: Driver chip model (e.g. "SN74HC04").
        receiver_chip: Receiver chip model (e.g. "SN74HCT04").
        document_ids: Document IDs to restrict search scope; empty means all documents.
        temperature: Operating temperature in Celsius for Layer 4 drift analysis.
    """
    driver_chip: Optional[str] = None
    receiver_chip: Optional[str] = None
    document_ids: List[str] = Field(default_factory=list)
    temperature: Optional[float] = None


def _get_erc_node(container) -> ERCNodeEnhanced:
    """Create a new ERCNodeEnhanced instance for this request.

    A fresh instance is created per request to prevent cross-request state
    pollution from the internal _param_cache.

    Args:
        container: ServiceContainer with retriever, erc_engine, and other services.

    Returns:
        A new ERCNodeEnhanced instance.

    Raises:
        HTTPException: 503 if retriever or erc_engine is not initialized.
    """
    retriever = container.retriever
    erc_engine = container.erc_engine

    if retriever is None or erc_engine is None:
        raise HTTPException(
            status_code=503,
            detail="ERC系统未初始化：检索器/引擎必须可用"
        )

    logger.info("Using enhanced ERC node")
    return ERCNodeEnhanced(
        retriever,
        erc_engine,
        settings=container.settings,
        llm_client=container.llm_client,
        enable_knowledge_graph=True,
        table_store=container.table_store,
    )


def _create_erc_metadata(document_ids: List[str]) -> MetadataState:
    """Create metadata state for ERC check.

    Args:
        document_ids: Document IDs to restrict search scope.

    Returns:
        MetadataState with selected_document_ids.
    """
    return MetadataState(
        selected_document_ids=document_ids or [],
    )


def _create_erc_state(query: str, document_ids: List[str]) -> AgentState:
    """Create initial AgentState for the ERC check pipeline.

    Args:
        query: Query string, typically "{driver} {receiver} ERC兼容性检查".
        document_ids: Document IDs to restrict search scope.

    Returns:
        AgentState initialized with query and metadata.
    """
    metadata = _create_erc_metadata(document_ids)
    return create_initial_state(query=query, metadata=metadata)


def _normalize_to_dict(obj: Union[Dict, Any]) -> Dict:
    """Normalize an object to dict format (adapter pattern).

    Handles dict, Pydantic BaseModel (model_dump), and dataclass/regular
    objects (__dict__) uniformly.

    Args:
        obj: Object to convert — dict, BaseModel, dataclass, or any object.

    Returns:
        Dict representation, or empty dict if conversion fails.
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return {}


def _extract_layer_results(erc_result: Union[Dict, Any]) -> List[Dict]:
    """Flatten four-layer ERC results into a single rule list.

    Args:
        erc_result: Four-layer ERC result (dict or FourLayerERCResult dataclass).

    Returns:
        Flat list of rule result dicts from all four layers.
    """
    result_dict = _normalize_to_dict(erc_result)
    rule_results = []

    for layer_key in LAYER_KEYS:
        layer_result = result_dict.get(layer_key)
        if not layer_result:
            continue

        layer_dict = _normalize_to_dict(layer_result)
        layer_rules = layer_dict.get("results", [])
        if layer_rules:
            rule_results.extend(layer_rules)

    return rule_results


def _get_severity_lower(rule_dict: Dict) -> str:
    """Get normalized lowercase severity string.

    Handles both string values ("ERROR") and Enum values (ERCSeverity.ERROR).

    Args:
        rule_dict: Rule result dict with "severity" field.

    Returns:
        Lowercase severity string: "error", "warning", or "info".
    """
    severity = rule_dict.get("severity", "")
    if isinstance(severity, str):
        return severity.lower()
    return str(getattr(severity, 'value', severity)).lower()


def _process_rule_entry(rule: Union[Dict, Any]) -> Optional[Dict]:
    """Convert a rule result to frontend entry format.

    Maps domain model fields to view model fields:
        rule_name → title, rule_id → type, message → description

    Args:
        rule: Single rule result (dict or ERCRuleResult dataclass).

    Returns:
        Frontend entry dict, or None if the rule is invalid.
    """
    rule_dict = _normalize_to_dict(rule)
    if not rule_dict:
        return None

    return {
        "title": rule_dict.get("rule_name", "未命名规则"),
        "type": rule_dict.get("rule_id", ""),
        "description": rule_dict.get("message", ""),
    }


def _categorize_rules(rule_results: List[Union[Dict, Any]]) -> Tuple[
    List[Dict], List[Dict], List[Dict],
    List[Dict], List[Dict], List[Dict]
]:
    """Categorize rules into errors, warnings, and info.

    Classification logic:
        passed=True  → info (check passed)
        passed=False + severity="warning" → warning (potential risk)
        passed=False + other severity → error (incompatible)

    Returns both raw rule dicts and frontend entry formats for each category.

    Args:
        rule_results: Flat list of rule results from _extract_layer_results.

    Returns:
        Tuple of (errors, warnings, info, error_entries, warning_entries, info_entries).
    """
    errors, warnings, info = [], [], []
    error_entries, warning_entries, info_entries = [], [], []

    for rule in rule_results:
        rule_dict = _normalize_to_dict(rule)
        if not rule_dict:
            continue

        entry = _process_rule_entry(rule)
        if entry is None:
            continue

        severity_lower = _get_severity_lower(rule_dict)
        passed = rule_dict.get("passed", False)

        if passed:
            info.append(rule_dict)
            info_entries.append(entry)
        elif severity_lower == "warning":
            warnings.append(rule_dict)
            warning_entries.append(entry)
        else:
            errors.append(rule_dict)
            error_entries.append(entry)

    return errors, warnings, info, error_entries, warning_entries, info_entries


def _build_erc_response(
    erc_result: Union[Dict, Any],
    state: AgentState,
    rule_results: List[Dict]
) -> Dict:
    """Build the full ERC check success response.

    Args:
        erc_result: Four-layer ERC result.
        state: AgentState containing extracted parameters and metadata.
        rule_results: Flat list of rule results.

    Returns:
        Complete response dict for the frontend.
    """
    result_dict = _normalize_to_dict(erc_result)

    overall_compatible = result_dict.get("overall_compatible", False)
    overall_confidence = result_dict.get("overall_confidence", 0.0)
    summary = result_dict.get("summary", "四层ERC检查完成")
    suggestions = result_dict.get("suggestions", [])

    errors, warnings, info, error_entries, warning_entries, info_entries = _categorize_rules(rule_results)

    driver_parameters = state.get("driver_parameters", {})
    receiver_parameters = state.get("receiver_parameters", {})
    driver_data_sources = state.get("driver_data_sources", {})
    receiver_data_sources = state.get("receiver_data_sources", {})
    processing_time = state.get("processing_time", 0.0)

    return {
        "success": True,
        "message": summary,
        "errors": error_entries,
        "warnings": warning_entries,
        "info": info_entries,
        "summary": summary,
        "erc_result": result_dict,
        "driver_parameters": driver_parameters,
        "receiver_parameters": receiver_parameters,
        "driver_data_sources": driver_data_sources,
        "receiver_data_sources": receiver_data_sources,
        "processing_time": processing_time,
        "overall_compatible": overall_compatible,
        "overall_confidence": overall_confidence,
        "suggestions": suggestions
    }


def _build_error_response(message: str, processing_time: float = 0.0) -> Dict:
    """Build a standardized error response.

    Uses a shallow copy of DEFAULT_ERROR_RESPONSE to avoid mutating the template.

    Args:
        message: Error description.
        processing_time: Elapsed processing time in seconds.

    Returns:
        Error response dict with the same structure as a success response.
    """
    response = DEFAULT_ERROR_RESPONSE.copy()
    response["message"] = message
    response["summary"] = message
    response["errors"] = [{"title": "ERC检查失败", "type": "ERROR", "description": message}]
    response["processing_time"] = processing_time
    return response


@router.post("/check")
async def erc_check(request: ERCCheckRequest, container=Depends(get_service_container)):
    """Run ERC compatibility check between driver and receiver chips.

    Validates input parameters, creates a fresh ERCNodeEnhanced instance,
    assembles the AgentState, executes the five-stage ERC pipeline, and
    returns categorized results (errors/warnings/info).

    Args:
        request: ERCCheckRequest with driver_chip, receiver_chip, document_ids, temperature.
        container: Injected ServiceContainer via FastAPI dependency.

    Returns:
        ERC check response dict with success status, categorized rules,
        extracted parameters, and overall compatibility conclusion.
    """
    driver_chip = (request.driver_chip or "").strip()
    receiver_chip = (request.receiver_chip or "").strip()

    if not driver_chip or not receiver_chip:
        raise HTTPException(status_code=400, detail="请提供驱动端与接收端芯片型号")

    try:
        erc_node = _get_erc_node(container)

        state = _create_erc_state(
            query=f"{driver_chip} {receiver_chip} ERC兼容性检查",
            document_ids=request.document_ids
        )
        logger.info(f"[API] ERC check request: driver={driver_chip}, receiver={receiver_chip}, document_ids={request.document_ids}")

        state["driver_chip"] = driver_chip
        state["receiver_chip"] = receiver_chip
        state["extracted_chips"] = [driver_chip, receiver_chip]
        state["intent"] = "erc_check"

        if request.temperature is not None:
            state["temperature"] = request.temperature

        state = await erc_node(state)

        erc_result = state.get("erc_result")

        if erc_result is None:
            error_msg = state.get("error") or "ERC检查失败"
            logger.error(f"ERC check failed: {error_msg}")
            return _build_error_response(error_msg, state.get("processing_time", 0.0))

        result_dict = _normalize_to_dict(erc_result)
        is_four_layer = any(k in result_dict for k in LAYER_KEYS)

        if is_four_layer:
            rule_results = _extract_layer_results(erc_result)
            return _build_erc_response(erc_result, state, rule_results)

        return {
            "success": True,
            "message": "ERC检查完成",
            "erc_result": result_dict,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ERC check exception: {str(e)}", exc_info=True)
        return _build_error_response(f"未知错误: {str(e)}")
