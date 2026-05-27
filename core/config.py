"""
Central configuration management for the VeriQuery system.

All modules import settings via ``from core.config import get_settings``.
Configuration priority: environment variables > .env file > code defaults.

Key design patterns:
  - Pydantic BaseSettings: automatic .env parsing, type coercion, and validation.
  - Singleton via lru_cache: ``get_settings()`` always returns the same instance.
  - Lazy properties: path configs use string fields + @property for Path objects.
"""

from typing import List, Dict
from pathlib import Path
from functools import lru_cache
import logging

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Global configuration for the VeriQuery system.

    Configuration groups:
        1. Application basics — version, debug mode
        2. API — host, port, CORS
        3. LLM — large language model path, device, quantization
        4. VLM — vision-language model for diagram understanding
        5. CLIP — image-text matching model for visual retrieval
        6. PDF processing — render DPI, max image size
        7. Text embedding — embedding model and dimension
        8. Vector database — ChromaDB storage
        9. Storage paths — data directories
        10. Retrieval — hybrid search weights and limits
        11. Chunking — text splitting parameters
        12. Chunk limits — prevent memory exhaustion
        13. Batching — GPU batch sizes
        14. SVG rendering — pinout diagram parameters
        15. Device search — Chinese-to-English keyword mappings
        16. PyTorch CUDA — GPU memory management
    """

    APP_VERSION: str = Field(default="3.1.0", description="Application version")
    DEBUG: bool = Field(default=False, description="Debug mode")

    API_HOST: str = Field(default="0.0.0.0", description="API listen address")
    API_PORT: int = Field(default=8000, description="API listen port")
    CORS_ORIGINS: List[str] = Field(default=["*"], description="CORS allowed origins")

    USE_HUGGINGFACE: bool = Field(default=True, description="Use HuggingFace local models")
    LLM_MODEL: str = Field(
        default="Qwen/Qwen2.5-1.5B",
        description="HuggingFace LLM model ID or local path (auto-downloaded on first run)",
    )
    LLM_DEVICE: str = Field(default="cuda", description="LLM device")
    LLM_QUANTIZE: bool = Field(default=True, description="Quantize LLM")
    LLM_QUANTIZATION_BITS: int = Field(default=4, description="LLM quantization bits (4/8)")
    LLM_TEMPERATURE: float = Field(default=0.3, description="LLM temperature")
    LLM_MAX_TOKENS: int = Field(default=256, description="LLM max output tokens")

    VLM_MODEL: str = Field(
        default="Qwen/Qwen2-VL-2B-Instruct",
        description="Vision-language model ID or local path (auto-downloaded on first run)",
    )
    VLM_QUANTIZE: bool = Field(default=True, description="Quantize VLM")
    VLM_QUANTIZATION_BITS: int = Field(default=4, description="VLM quantization bits (4/8)")

    CLIP_MODEL: str = Field(
        default="openai/clip-vit-base-patch32",
        description="CLIP model ID or local path (auto-downloaded on first run)",
    )
    CLIP_THRESHOLD: float = Field(default=0.25, description="CLIP similarity threshold (0-1)")

    PDF_RENDER_DPI: int = Field(default=200, description="PDF render DPI")
    MAX_IMAGE_SIZE: int = Field(default=896, description="Max image size in pixels")

    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-large-zh-v1.5",
        description="Text embedding model ID or local path (auto-downloaded on first run)",
    )
    EMBEDDING_DEVICE: str = Field(default="cuda", description="Embedding model device")
    EMBEDDING_DIMENSION: int = Field(default=1024, description="Embedding vector dimension")

    CHROMA_PERSIST_DIR: str = Field(default="./data/chroma", description="ChromaDB persist directory")
    CHROMA_COLLECTION_NAME: str = Field(default="veriquery_text", description="ChromaDB collection name")

    BASE_DIR_STR: str = Field(default="", description="Project root directory")
    DATA_DIR_STR: str = Field(default="", description="Data root directory")
    UPLOAD_DIR_STR: str = Field(default="", description="Upload directory")
    PROCESSED_DIR_STR: str = Field(default="", description="Processed data directory")
    IMAGE_DIR_STR: str = Field(default="", description="Image storage directory")
    CACHE_DIR_STR: str = Field(default="", description="Cache directory")

    VECTOR_WEIGHT: float = Field(default=0.5, description="Vector search weight")
    BM25_WEIGHT: float = Field(default=0.35, description="BM25 search weight")
    STRUCTURED_WEIGHT: float = Field(default=0.15, description="Structured search weight")
    MAX_RESULTS_PER_DOC: int = Field(default=5, description="Max results per document")
    MAX_RESULTS_PER_DOC_CROSS_SOURCE: int = Field(
        default=8, description="Max results per document when hit by multiple sources"
    )

    CHUNK_SIZE: int = Field(default=800, description="Chunk size in characters")
    CHUNK_OVERLAP: int = Field(default=200, description="Chunk overlap in characters")

    MAX_CHUNKS_PER_PAGE: int = Field(default=100, description="Max chunks per page")
    MAX_TOTAL_CHUNKS: int = Field(default=100000, description="Max total chunks per document")

    BATCH_SIZE_EMBEDDING: int = Field(default=8, description="Embedding batch size")
    BATCH_SIZE_INDEXING: int = Field(default=10, description="Indexing batch size (pages/batch)")
    MIN_BATCH_SIZE: int = Field(default=1, description="Min batch size")
    MAX_BATCH_SIZE: int = Field(default=8, description="Max batch size")

    SVG_PIN_WIDTH: int = Field(default=240, description="Pin rectangle width")
    SVG_PIN_HEIGHT: int = Field(default=62, description="Pin rectangle height")
    SVG_PIN_SPACING: int = Field(default=10, description="Pin spacing")
    SVG_CHIP_PADDING: int = Field(default=300, description="Chip body padding")
    SVG_PIN_NUMBER_FONT_SIZE: int = Field(default=17, description="Pin number font size")
    SVG_PIN_NAME_FONT_SIZE: int = Field(default=20, description="Pin name font size")
    SVG_TITLE_FONT_SIZE: int = Field(default=30, description="Title font size")
    SVG_SHOW_LEGEND: bool = Field(default=True, description="Show legend in pinout diagram")
    SVG_DEFAULT_PACKAGE: str = Field(default="DIP-8", description="Default package type")

    DEVICE_SEARCH_KEYWORD_MAPPINGS: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "反相器": ["inverter", "not", "74"],
            "逻辑门": ["gate", "74"],
            "缓冲器": ["buffer", "74"],
            "触发器": ["flipflop", "74", "flip-flop"],
            "计数器": ["counter", "74"],
            "寄存器": ["register", "74"],
            "运算放大器": ["op", "amp", "amplifier", "operational"],
            "比较器": ["comparator"],
            "稳压器": ["regulator", "ldo", "dc-dc"],
            "转换器": ["converter", "adc", "dac"],
            "存储器": ["memory", "flash", "ram", "rom", "eeprom"],
            "微控制器": ["mcu", "microcontroller", "stm32", "arduino"],
            "传感器": ["sensor"],
            "接口": ["interface", "uart", "spi", "i2c", "can", "usb"],
        },
        description="Chinese-to-English keyword mappings for device search",
    )

    PYTORCH_CUDA_ALLOC_CONF: str = Field(
        default="max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.8",
        description="PyTorch CUDA memory allocator config",
    )

    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False
        extra = "ignore"

    @property
    def BASE_DIR(self) -> Path:
        """Project root directory. Derived from config.py location if not set."""
        if self.BASE_DIR_STR:
            return Path(self.BASE_DIR_STR)
        return Path(__file__).parent.parent

    @property
    def DATA_DIR(self) -> Path:
        """Data root directory (default: BASE_DIR/data)."""
        if self.DATA_DIR_STR:
            return Path(self.DATA_DIR_STR)
        return self.BASE_DIR / "data"

    @property
    def UPLOAD_DIR(self) -> Path:
        """Upload directory (default: DATA_DIR/uploads)."""
        if self.UPLOAD_DIR_STR:
            return Path(self.UPLOAD_DIR_STR)
        return self.DATA_DIR / "uploads"

    @property
    def PROCESSED_DIR(self) -> Path:
        """Processed data directory (default: DATA_DIR/processed)."""
        if self.PROCESSED_DIR_STR:
            return Path(self.PROCESSED_DIR_STR)
        return self.DATA_DIR / "processed"

    @property
    def IMAGE_DIR(self) -> Path:
        """Image storage directory (default: DATA_DIR/images)."""
        if self.IMAGE_DIR_STR:
            return Path(self.IMAGE_DIR_STR)
        return self.DATA_DIR / "images"

    @property
    def CACHE_DIR(self) -> Path:
        """Cache directory (default: DATA_DIR/cache)."""
        if self.CACHE_DIR_STR:
            return Path(self.CACHE_DIR_STR)
        return self.DATA_DIR / "cache"

    @property
    def VECTOR_DB_DIR(self) -> Path:
        """Vector database directory (default: DATA_DIR/vector_db)."""
        return self.DATA_DIR / "vector_db"

    def ensure_directories(self):
        """Create all required directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.UPLOAD_DIR,
            self.PROCESSED_DIR,
            self.IMAGE_DIR,
            self.CACHE_DIR,
            self.VECTOR_DB_DIR,
        ]
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def validate_config(self):
        """Validate configuration values and raise ValueError on invalid settings.

        Checks:
            - API_PORT in [1024, 65535]
            - LLM_TEMPERATURE in [0, 2.0]
            - Search weights in [0, 1.0] and sum to 1.0 (±0.01)
            - Batch sizes > 0
            - CHUNK_SIZE > 0, CHUNK_OVERLAP >= 0, CHUNK_OVERLAP < CHUNK_SIZE
            - Chunk limits > 0
        """
        errors = []

        if not (1024 <= self.API_PORT <= 65535):
            errors.append(f"API_PORT {self.API_PORT} out of range (1024-65535)")

        if not (0 <= self.LLM_TEMPERATURE <= 2.0):
            errors.append(f"LLM_TEMPERATURE {self.LLM_TEMPERATURE} out of range (0-2.0)")

        if not (0 <= self.VECTOR_WEIGHT <= 1.0):
            errors.append(f"VECTOR_WEIGHT {self.VECTOR_WEIGHT} out of range (0-1.0)")

        if not (0 <= self.BM25_WEIGHT <= 1.0):
            errors.append(f"BM25_WEIGHT {self.BM25_WEIGHT} out of range (0-1.0)")

        if not (0 <= self.STRUCTURED_WEIGHT <= 1.0):
            errors.append(f"STRUCTURED_WEIGHT {self.STRUCTURED_WEIGHT} out of range (0-1.0)")

        weight_sum = self.VECTOR_WEIGHT + self.BM25_WEIGHT + self.STRUCTURED_WEIGHT
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(
                f"Search weights sum {weight_sum:.2f} != 1.0 "
                f"(VECTOR={self.VECTOR_WEIGHT} + BM25={self.BM25_WEIGHT} "
                f"+ STRUCTURED={self.STRUCTURED_WEIGHT})"
            )

        if self.BATCH_SIZE_EMBEDDING <= 0:
            errors.append(f"BATCH_SIZE_EMBEDDING {self.BATCH_SIZE_EMBEDDING} must be > 0")

        if self.BATCH_SIZE_INDEXING <= 0:
            errors.append(f"BATCH_SIZE_INDEXING {self.BATCH_SIZE_INDEXING} must be > 0")

        if self.CHUNK_SIZE <= 0:
            errors.append(f"CHUNK_SIZE {self.CHUNK_SIZE} must be > 0")

        if self.CHUNK_OVERLAP < 0:
            errors.append(f"CHUNK_OVERLAP {self.CHUNK_OVERLAP} must be >= 0")

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            errors.append(
                f"CHUNK_OVERLAP {self.CHUNK_OVERLAP} must be < CHUNK_SIZE {self.CHUNK_SIZE}"
            )

        if self.MAX_CHUNKS_PER_PAGE <= 0:
            errors.append(f"MAX_CHUNKS_PER_PAGE {self.MAX_CHUNKS_PER_PAGE} must be > 0")

        if self.MAX_TOTAL_CHUNKS <= 0:
            errors.append(f"MAX_TOTAL_CHUNKS {self.MAX_TOTAL_CHUNKS} must be > 0")

        if errors:
            error_msg = "\n".join(errors)
            raise ValueError(f"Configuration validation failed:\n{error_msg}")

        logger = logging.getLogger(__name__)
        logger.info("Configuration validation passed")


@lru_cache()
def get_settings() -> Settings:
    """Return the global Settings singleton.

    Uses lru_cache for thread-safe singleton behavior.
    Call ``get_settings.cache_clear()`` to reset (e.g. in tests).
    """
    return Settings()
