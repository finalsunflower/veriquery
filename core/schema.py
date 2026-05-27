"""
Unified state machine and data model definitions for VeriQuery.

This module is the data contract center of the system. All data structures
shared between modules are defined here — no module may define its own
private data formats.

Key components:
    - PinType / ERCSeverity / ExtractionSource: Enumerations
    - Citation / PinInfo / ElectricalSpec / RetrievedChunk: Pydantic DTOs
    - MetadataState / AgentState: LangGraph TypedDict state definitions
    - create_initial_state: Factory function for workflow entry
"""

from typing import TypedDict, List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class PinType(str, Enum):
    """Pin type enumeration for visualization color-coding and ERC classification."""

    POWER = "power"
    GROUND = "ground"
    IO = "io"
    BIDIRECTIONAL = "bidirectional"
    INPUT = "input"
    OUTPUT = "output"
    ANALOG = "analog"
    NC = "nc"
    SPECIAL = "special"


class ERCSeverity(str, Enum):
    """ERC check result severity level."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ExtractionSource(str, Enum):
    """Parameter extraction source enumeration for confidence assessment.

    The three-stage extraction pipeline produces results from different
    sources with varying reliability:
        TABLE  — structured PDF table extraction (highest confidence)
        REGEX  — regex pattern matching from text (medium confidence)
        LLM   — LLM-based extraction from unstructured text (lowest confidence)
    """

    TABLE = "table"
    REGEX = "regex"
    LLM = "llm"


class Citation(BaseModel):
    """Citation source data model for traceability and verification.

    Every factual statement in the system must carry a citation so that
    users can verify the answer against the original document.

    Attributes:
        file: Source filename, e.g. 'STM32F103.pdf'.
        page: Page number (1-indexed).
        text_snippet: Quoted text fragment from the source.
        char_start: Character offset for precise highlight positioning.
        char_end: Character offset end position.
    """

    file: str = Field(..., description="Source filename")
    page: int = Field(..., ge=1, description="Page number (1-indexed)")
    text_snippet: str = Field(default="", description="Quoted text fragment")
    char_start: int = Field(default=0, ge=0, description="Character offset start")
    char_end: int = Field(default=0, ge=0, description="Character offset end")


class PinInfo(BaseModel):
    """Pin information data model for pinout visualization and ERC.

    Attributes:
        number: Pin number (1-indexed).
        name: Pin name, e.g. 'PA0', 'VCC', 'GND'.
        pin_type: Pin type enum, defaults to IO.
        functions: List of alternate functions, e.g. ['GPIO', 'ADC_IN0'].
        electrical_level: Electrical characteristic label, e.g. '5V tolerant'.
    """

    number: int = Field(..., ge=1, description="Pin number (1-indexed)")
    name: str = Field(..., min_length=1, description="Pin name")
    pin_type: PinType = Field(default=PinType.IO, description="Pin type")
    functions: List[str] = Field(default_factory=list, description="Alternate function list")
    electrical_level: str = Field(default="", description="Electrical level label, e.g. '5V tolerant'")

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """Validate pin name is not empty after stripping whitespace."""
        v = v.strip()
        if not v:
            raise ValueError("引脚名称不能为空")
        return v

    @field_validator("functions", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> List[str]:
        """Convert slash-separated string to list, e.g. 'GPIO/ADC/TIM' → ['GPIO','ADC','TIM']."""
        if isinstance(v, str):
            return [f.strip() for f in v.split("/") if f.strip()]
        return v if isinstance(v, list) else []


class ElectricalSpec(BaseModel):
    """Electrical parameter specification model.

    Represents a complete electrical parameter with min/typ/max values,
    matching the "Electrical Characteristics" table in datasheets.

    Attributes:
        param: Parameter symbol, e.g. 'VOH', 'VIL', 'II'.
        name: Full parameter name, e.g. 'Output High Voltage'.
        min_value: Minimum value (optional, not all specs have all three).
        typ_value: Typical value (optional).
        max_value: Maximum value (optional).
        unit: Unit string, e.g. 'V', 'mA', 'MHz'.
        condition: Test condition, e.g. 'IOH=-4mA, VCC=3.3V'.
        citation: Source citation for traceability.
        source_type: Extraction source type for confidence assessment.
        confidence: Extraction confidence score in [0, 1].
    """

    param: str = Field(..., description="Parameter symbol")
    name: str = Field(default="", description="Full parameter name")
    min_value: Optional[float] = Field(default=None, description="Minimum value")
    typ_value: Optional[float] = Field(default=None, description="Typical value")
    max_value: Optional[float] = Field(default=None, description="Maximum value")
    unit: str = Field(default="", description="Unit, e.g. 'V', 'mA', 'MHz'")
    condition: str = Field(default="", description="Test condition")
    citation: Optional[Citation] = Field(default=None, description="Source citation")
    source_type: ExtractionSource = Field(default=ExtractionSource.TABLE, description="Extraction source type")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Confidence score [0, 1]")


class RetrievedChunk(BaseModel):
    """Retrieved text chunk — unified output of the hybrid retriever.

    Regardless of retrieval source (dense vector, BM25 sparse, table, image),
    all results are normalized to this format for downstream consumption
    and RRF (Reciprocal Rank Fusion) merging.

    Attributes:
        chunk_id: Unique chunk identifier for deduplication during RRF fusion.
        text: Raw text content for LLM context.
        score: Relevance score normalized to [0, 1].
        document_id: Parent document ID in the vector store.
        filename: Source filename for display.
        page: Page number (1-indexed).
        char_start: Character offset start for highlight positioning.
        char_end: Character offset end.
        source: Retrieval source tag: dense/sparse/table/image/hybrid.
        metadata: Additional metadata dict (original scores, retrieval details).
    """

    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Text content")
    score: float = Field(default=0.0, ge=0, le=1, description="Relevance score [0, 1]")
    document_id: str = Field(..., description="Parent document ID")
    filename: str = Field(default="", description="Source filename")
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    char_start: int = Field(default=0, ge=0, description="Character offset start")
    char_end: int = Field(default=0, ge=0, description="Character offset end")
    source: str = Field(default="hybrid", description="Retrieval source tag")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MetadataState(TypedDict):
    """Metadata state — stores the current session's document context.

    Attributes:
        selected_document_ids: User-selected document IDs for scoped retrieval.
            Empty list means search all documents.
    """

    selected_document_ids: List[str]


class AgentState(TypedDict):
    """LangGraph unified state machine — the central state hub for all workflow nodes.

    Each node receives the current state, returns a dict of field updates,
    and LangGraph automatically merges them. Fields are organized in layers:

        Input layer:      query
        Metadata layer:   metadata, user_context
        Routing layer:    intent, extracted_chips
        Extraction layer: extracted_data
        Reasoning layer:  erc_result, comparison_matrix
        Output layer:     final_response, citations
        Control layer:    error, processing_time

    Attributes:
        query: User's original query string.
        metadata: Document metadata with selected_document_ids.
        user_context: API-injected user context (session_id, preferences, etc.).
        intent: Recognized intent: 'qa', 'pinout', 'erc_check', 'comparison', or None.
        extracted_chips: Chip model names extracted from the query.
        extracted_data: Extraction results (structure varies by intent).
        erc_result: ERC check result dict with errors/warnings/info lists, or None.
        comparison_matrix: Multi-chip comparison result dict, or None.
        final_response: Final response text generated by LLM or formatter.
        citations: Citation list (LangGraph append-type field).
        error: Error message string, or None if no error.
        processing_time: Processing duration in seconds.
    """

    query: str
    metadata: MetadataState
    user_context: Dict
    intent: Optional[str]
    extracted_chips: List[str]
    extracted_data: Dict
    erc_result: Optional[Dict]
    comparison_matrix: Optional[Dict]
    final_response: str
    citations: List[Dict]
    error: Optional[str]
    processing_time: float


def create_initial_state(
    query: str,
    metadata: Optional[MetadataState] = None,
) -> AgentState:
    """Create an initial AgentState with all fields set to default values.

    This factory function ensures completeness — every field has an initial
    value so that no KeyError occurs during workflow execution.

    Args:
        query: User query string.
        metadata: Optional document metadata; defaults to empty MetadataState
            (searches all documents).

    Returns:
        AgentState with all fields initialized.
    """
    return AgentState(
        query=query,
        metadata=metadata or MetadataState(
            selected_document_ids=[],
        ),
        user_context={},
        intent=None,
        extracted_chips=[],
        extracted_data={},
        erc_result=None,
        comparison_matrix=None,
        final_response="",
        citations=[],
        error=None,
        processing_time=0.0,
    )
