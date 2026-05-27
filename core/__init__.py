"""
VeriQuery 核心模块 - 统一数据契约

【重要】这是整个系统的数据契约中心。
所有其他模块必须从此处导入数据类型，严禁私自定义。

使用示例:
    from core import AgentState, PinInfo, ElectricalSpec, Citation
    from core import Settings, get_settings
"""

# ===== 数据模型 =====
from .schema import (
    PinType,
    ERCSeverity,
    ExtractionSource,
    Citation,
    PinInfo,
    ElectricalSpec,
    RetrievedChunk,
    MetadataState,
    AgentState,
    create_initial_state,
)

# ===== 配置管理 =====
from .config import (
    Settings,
    get_settings,
)

# ===== 异常定义 =====
from .exceptions import (
    VeriQueryError,
    RetrievalError,
    ConfigurationError,
    ProcessingError,
)

# ===== 清理管理器 =====
from .cleanup_manager import CleanupManager, create_cleanup_manager

# ===== 模型管理器 =====
from .model_manager import model_manager

# ===== SVG 渲染器 =====
from .svg_renderer import PinoutSVGRenderer, SVGConfig

__version__ = "3.1.0"
__author__ = "VeriQuery Team"

__all__ = [
    "__version__",
    "PinType", 
    "ERCSeverity",
    "ExtractionSource",
    "Citation",
    "PinInfo",
    "ElectricalSpec",
    "RetrievedChunk",
    "MetadataState",
    "AgentState",
    "create_initial_state",
    "Settings",
    "get_settings",
    "VeriQueryError",
    "RetrievalError",
    "ConfigurationError",
    "ProcessingError",
    "CleanupManager",
    "create_cleanup_manager",
    "model_manager",
    "PinoutSVGRenderer",
    "SVGConfig",
]