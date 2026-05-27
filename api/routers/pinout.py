"""
Pinout diagram router.

Provides API endpoints for querying chip pin information and generating
SVG pinout diagrams. Uses an Agent workflow (LangGraph) to retrieve and
extract pin data from uploaded documents, then renders visual pinout
diagrams via the core SVG renderer.

Endpoints:
  POST /pinout/  → Query pin information and generate pinout diagram
"""

import logging
from typing import List, Optional, Dict, Any, Set, Tuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core import PinInfo as CorePinInfo, PinType, get_settings
from knowledge.pinout_library import CommonPinoutLibrary
from api.dependencies import get_service_container

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/pinout", tags=["pinout"])


class PinoutRequest(BaseModel):
    """Request body for pinout query.

    Attributes:
        chip_name: Target chip model (e.g. "STM32F103C8T6").
        document_ids: Document IDs to restrict search scope; empty means all documents.
        package: Package type (e.g. "DIP-8", "LQFP48"); None for auto-detection.
    """
    chip_name: str
    document_ids: List[str] = Field(default_factory=list)
    package: Optional[str] = None


class PinInfo(BaseModel):
    """Single pin information for API response.

    Maps domain model (CorePinInfo) fields to a frontend-friendly format.
    CorePinInfo is used by the SVG renderer; PinInfo is used in the HTTP
    response with additional display fields.

    Attributes:
        pin: Pin number (1-indexed).
        name: Pin name (e.g. "PA0", "VCC", "GND").
        pin_type: Pin type string — one of PinType enum values.
        functions: Primary and alternate function list.
        alternate_functions: Lower-priority alternate functions.
        electrical: Electrical characteristics dict (e.g. {"level": "5V tolerant"}).
        description: Pin description text.
    """
    pin: int
    name: str
    pin_type: str = "io"
    functions: List[str] = Field(default_factory=list)
    alternate_functions: List[str] = Field(default_factory=list)
    electrical: dict = Field(default_factory=dict)
    description: str = ""


class PinoutResponse(BaseModel):
    """Pinout query response.

    Attributes:
        success: Whether the query succeeded.
        chip_name: Queried chip name.
        package: Resolved package type (user-specified > extracted > default).
        pin_count: Total number of pins.
        pins: Detailed pin information list.
        svg: SVG pinout diagram string.
        citations: Source citations for traceability.
        available_packages: Available package types for this chip.
    """
    success: bool = True
    chip_name: str
    package: str = ""
    pin_count: int = 0
    pins: List[PinInfo] = Field(default_factory=list)
    svg: str = ""
    citations: List[dict] = Field(default_factory=list)
    available_packages: List[str] = Field(default_factory=list)


def _extract_available_packages(extracted: Dict[str, Any]) -> List[str]:
    """Extract available package types from agent-extracted data.

    Two-level fallback:
        1. extracted["available_packages"] — direct list from LLM
        2. extracted["package"] or extracted["package_type"] — single package

    Args:
        extracted: Agent workflow extracted data dict.

    Returns:
        List of available package type strings, or empty list.
    """
    available_packages = extracted.get("available_packages")
    if available_packages:
        return list(available_packages)

    packages_set: Set[str] = set()
    extracted_package = extracted.get("package") or extracted.get("package_type")
    if extracted_package:
        packages_set.add(extracted_package)

    return list(packages_set)


_PIN_TYPE_ALIASES: Dict[str, str] = {
    "bi-directional": "bidirectional",
    "bidir": "bidirectional",
    "gpio": "bidirectional",
    "in": "input",
    "out": "output",
    "gnd": "ground",
    "pwr": "power",
    "vcc": "power",
    "vdd": "power",
    "vss": "ground",
    "analog_in": "analog",
    "analog_out": "analog",
    "adc": "analog",
    "dac": "analog",
    "no_connect": "nc",
    "n.c.": "nc",
    "n/a": "nc",
    "reset": "special",
    "boot": "special",
    "clock": "special",
    "clk": "special",
    "xtal": "special",
    "osc": "special",
    "i/o": "io",
    "io": "io",
}


def _normalize_pin_type(raw_type: str) -> str:
    """Normalize pin type string to a PinType enum value.

    Looks up the alias table first; if not found, assumes the value is
    already a valid PinType member and returns it as-is.

    Args:
        raw_type: Raw pin type string (already lowercased and stripped).

    Returns:
        Normalized pin type string.
    """
    return _PIN_TYPE_ALIASES.get(raw_type, raw_type)


def _create_pin_objects(p: Dict[str, Any]) -> Tuple[CorePinInfo, PinInfo]:
    """Create both CorePinInfo and PinInfo from a raw pin dict.

    CorePinInfo is used by the SVG renderer; PinInfo is used in the API
    response. Creating both in one pass avoids a second traversal.

    Args:
        p: Raw pin dict from LLM extraction, with keys like "number"/"pin",
           "name", "pin_type", "functions", etc.

    Returns:
        Tuple of (CorePinInfo, PinInfo).

    Raises:
        ValueError: If pin number, name, or type is missing or invalid.
    """
    number_raw = p.get("number") or p.get("pin")
    if number_raw is None:
        raise ValueError("引脚编号缺失，无法生成引脚图")

    try:
        pin_number = int(number_raw)
    except (ValueError, TypeError) as e:
        raise ValueError(f"引脚编号格式无效: '{number_raw}'，无法转换为整数") from e

    pin_name = p.get("name")
    if pin_name is None:
        raise ValueError(f"引脚 {pin_number} 名称缺失，无法生成引脚图")

    cleaned_name = str(pin_name).strip()
    if not cleaned_name:
        raise ValueError(f"引脚 {pin_number} 名称为空，无法生成引脚图")

    pin_type_value = p.get("pin_type")
    if not pin_type_value:
        raise ValueError(f"引脚 {pin_number} 类型缺失，无法生成引脚图")

    pin_type_normalized = _normalize_pin_type(str(pin_type_value).strip().lower())

    try:
        pin_type = PinType(pin_type_normalized)
    except ValueError as e:
        raise ValueError(f"引脚 {pin_number} 类型无效: '{pin_type_value}'") from e

    electrical_level = p.get("electrical_level", "") or ""
    functions = p.get("functions") or []
    alternate_functions = p.get("alternate_functions") or []
    description = p.get("description", "") or ""

    if not alternate_functions and len(functions) > 1:
        alternate_functions = [f for f in functions if f.upper() != cleaned_name.upper()]

    core_pin = CorePinInfo(
        number=pin_number,
        name=cleaned_name,
        pin_type=pin_type,
        functions=functions,
        electrical_level=electrical_level
    )

    response_pin = PinInfo(
        pin=pin_number,
        name=cleaned_name,
        pin_type=pin_type.value,
        functions=functions,
        alternate_functions=alternate_functions,
        electrical={"level": electrical_level} if electrical_level else {},
        description=description
    )

    return core_pin, response_pin


def _parse_pin_data(pinout_list: List[Dict[str, Any]]) -> Tuple[List[CorePinInfo], List[PinInfo]]:
    """Parse a list of raw pin dicts into CorePinInfo and PinInfo lists.

    Uses fail-fast strategy: if any pin is invalid, the entire parse
    aborts with a ValueError. Partial pinout diagrams are more misleading
    than no diagram at all.

    Args:
        pinout_list: List of raw pin dicts from LLM extraction.

    Returns:
        Tuple of (core_pins, response_pins). Returns ([], []) for empty input.
    """
    if not pinout_list:
        return [], []

    core_pins: List[CorePinInfo] = []
    response_pins: List[PinInfo] = []

    for p in pinout_list:
        core_pin, response_pin = _create_pin_objects(p)
        core_pins.append(core_pin)
        response_pins.append(response_pin)

    return core_pins, response_pins


@router.post("/", response_model=PinoutResponse)
async def get_pinout(request: PinoutRequest, container=Depends(get_service_container)):
    """Query chip pin information and generate pinout diagram.

    Executes the Agent workflow to retrieve and extract pin data from
    documents, resolves the package type, renders an SVG diagram, and
    returns the complete pinout response.

    Args:
        request: PinoutRequest with chip_name, document_ids, and optional package.
        container: Injected ServiceContainer via FastAPI dependency.

    Returns:
        PinoutResponse with pin list, SVG diagram, citations, and package info.

    Raises:
        HTTPException: 400 for invalid input, 404 if no pin data found,
            422 for unparseable data, 500 for server errors, 503 if services
            are unavailable.
    """
    chip_name = request.chip_name.strip() if request.chip_name else ""
    if not chip_name:
        raise HTTPException(status_code=400, detail="芯片名称不能为空")

    graph = container.veriquery_graph
    if graph is None:
        raise HTTPException(status_code=503, detail="VeriQuery图未初始化 - Agent系统不可用")

    if not container.svg_renderer:
        raise HTTPException(status_code=503, detail="SVG渲染器未初始化 - 无法生成引脚图")

    try:
        result_state = await graph.ainvoke(
            question=f"列出 {chip_name} 的所有引脚定义、引脚编号和封装类型",
            session_id="pinout_query",
            user_context={
                "selected_document_ids": request.document_ids,
            }
        )

        extracted = result_state.get("extracted_data") or {}
        pinout_list = extracted.get("pinout") or []

        if not pinout_list:
            raise HTTPException(
                status_code=404,
                detail=f"未能从文档中提取到 {chip_name} 的引脚信息 - 请检查文档质量和检索系统"
            )

        try:
            core_pins, response_pins = _parse_pin_data(pinout_list)
        except ValueError as ve:
            raise HTTPException(status_code=422, detail=f"引脚数据解析失败: {str(ve)}")

        if not core_pins:
            raise HTTPException(
                status_code=422,
                detail=f"解析 {chip_name} 引脚数据失败 - 所有引脚数据都无效"
            )

        available_packages = _extract_available_packages(extracted)

        extracted_package = extracted.get("package") or extracted.get("package_type")
        package = request.package or extracted_package

        if not package:
            std_pinout = CommonPinoutLibrary.get_pinout(chip_name)
            std_package = std_pinout.get("package") if std_pinout else None
            if std_package:
                package = std_package
                logger.info("封装类型从标准引脚库获取: %s -> %s", chip_name, package)
            else:
                package = settings.SVG_DEFAULT_PACKAGE
                logger.warning("封装类型未从文档中提取，使用默认封装 %s", package)

        svg = container.svg_renderer.render(core_pins, package=package, chip_name=chip_name)

        if not svg:
            raise HTTPException(status_code=500, detail="SVG引脚图生成失败 - 渲染器返回空结果")

        logger.info("引脚图生成完成: chip=%s, pins=%d, svg_len=%d", chip_name, len(core_pins), len(svg))

        citations = result_state.get("citations") or []
        if not citations:
            extracted = result_state.get("extracted_data") or {}
            pinout_list = extracted.get("pinout") or []
            if pinout_list:
                citations = [{
                    "file": f"标准引脚库 - {chip_name}",
                    "page": 1,
                    "text_snippet": f"{chip_name} 引脚定义 ({len(pinout_list)}引脚, {package or '未知封装'})",
                    "confidence": 0.9,
                    "source": "standard_library",
                }]
                logger.info("为引脚分析生成合成引用: chip=%s, pins=%d", chip_name, len(pinout_list))
            else:
                logger.warning("引脚分析未提供引用信息且无引脚数据: chip=%s", chip_name)

        return PinoutResponse(
            success=True,
            chip_name=chip_name,
            package=package,
            pin_count=len(response_pins),
            pins=response_pins,
            svg=svg,
            citations=citations,
            available_packages=available_packages
        )

    except HTTPException:
        raise
    except ValueError as ve:
        logger.warning("引脚查询验证错误: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as err:
        logger.error("引脚查询运行时错误: %s", err)
        raise HTTPException(status_code=500, detail=f"系统运行时错误: {str(err)}")
    except Exception as e:
        logger.error("引脚查询系统错误: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"引脚查询系统故障，无法提供可靠结果: {str(e)}")
