"""
Device comparison router - Parameter comparison with multi-layer scoring.

Provides device parameter comparison using ComparisonNodeEnhanced with
three-layer scoring architecture (CCM + Z-A-FoM + B-SPOTIS).
"""

import logging
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core import create_initial_state
from api.dependencies import get_service_container
from agents.comparison_node import ComparisonNodeEnhanced

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compare", tags=["compare"])


class CompareEnhancedRequest(BaseModel):
    """Request model for enhanced device comparison."""

    devices: List[str]
    document_ids: List[str] = Field(default_factory=list)


class CompareEnhancedResponse(BaseModel):
    """Response model for enhanced device comparison.

    All optional fields have defaults to support graceful degradation
    when partial results are available.
    """

    success: bool = True
    devices: List[Dict[str, Any]]
    comparison_matrix: Dict[str, Any] = Field(default_factory=dict)
    scoring_result: Dict[str, Any] = Field(default_factory=dict)
    radar_data: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    processing_time: float = 0.0


@router.post("/devices-enhanced", response_model=CompareEnhancedResponse)
async def compare_devices_enhanced(
    request: CompareEnhancedRequest,
    container=Depends(get_service_container),
):
    """Compare device parameters using three-layer scoring (CCM + Z-A-FoM + B-SPOTIS).

    Flow: validate request → get dependencies → create agent → build state
          → execute comparison → extract results → return response.

    Args:
        request: CompareEnhancedRequest with device list and optional document IDs.
        container: ServiceContainer via dependency injection.

    Returns:
        CompareEnhancedResponse with comparison matrix, scoring, radar data, and summary.

    Raises:
        HTTPException 400: Invalid request parameters.
        HTTPException 503: Service dependencies unavailable.
        HTTPException 500: Internal server error.
    """
    try:
        if not request.devices:
            raise ValueError("设备列表不能为空")

        if len(request.devices) < 2:
            raise ValueError("至少需要2个设备进行比较")

        retriever = container.retriever
        if retriever is None:
            raise RuntimeError("检索器未初始化")

        comparison_node = ComparisonNodeEnhanced(
            retriever=retriever,
            settings=container.settings,
            llm_client=container.llm_client,
            table_store=container.table_store,
        )

        state = create_initial_state(
            query=f"对比分析 {', '.join(request.devices)} 的参数",
            metadata={
                "selected_document_ids": request.document_ids,
            },
        )

        state["chips"] = request.devices
        state["intent"] = "comparison"
        state["selected_document_ids"] = request.document_ids

        result_state = await comparison_node(state)

        if result_state.get("error"):
            raise RuntimeError(result_state["error"])

        return CompareEnhancedResponse(
            success=True,
            devices=[{"part_number": d} for d in request.devices],
            comparison_matrix=result_state.get("comparison_matrix", {}),
            scoring_result=result_state.get("scoring_result", {}),
            radar_data=result_state.get("radar_data", {}),
            summary=result_state.get("summary", ""),
            processing_time=result_state.get("processing_time", 0.0),
        )

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(f"设备比较参数错误: {exc}")
        raise HTTPException(status_code=400, detail=f"参数错误: {str(exc)}")

    except RuntimeError as exc:
        logger.error(f"设备比较运行时错误: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))

    except Exception as exc:
        logger.error(f"设备比较异常: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"设备比较系统错误: {str(exc)}")
