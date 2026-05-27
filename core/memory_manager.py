"""
GPU memory manager — monitoring, smart cleanup, and pre-load checks.

Provides real-time VRAM statistics, trend tracking, two-tier cleanup
(normal GC vs aggressive model unloading), and a thread-safe singleton
factory. Used by llm_client, cleanup_manager, embeddings, and API layer.
"""

import gc
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """GPU VRAM statistics snapshot.

    Attributes:
        available: Whether CUDA is usable. When False, other fields are 0.0.
        total_memory: Total GPU VRAM in GiB.
        used_memory: Used VRAM in GiB (total - free).
        free_memory: Free VRAM in GiB, from torch.cuda.mem_get_info().
        utilization_percent: VRAM usage percentage (0–100).
        device_name: GPU device name, e.g. "NVIDIA GeForce RTX 4090".
    """

    available: bool
    total_memory: float
    used_memory: float
    free_memory: float
    utilization_percent: float
    device_name: str


class MemoryManager:
    """GPU VRAM monitor with smart cleanup and pre-load checks.

    Cleanup strategies:
        - Normal: gc.collect() + torch.cuda.empty_cache() (< 1 second).
        - Aggressive: above + unload inactive models + second empty_cache().

    Trend tracking:
        Maintains a deque of the last 10 (timestamp, used_gb) samples to
        classify memory trend as increasing / decreasing / stable.

    Debounce:
        A 2-second cooldown between cleanups prevents thrashing.
    """

    def __init__(self):
        self._cuda_available = torch.cuda.is_available()
        self._device_name = "CPU"
        self._total_memory_gb = 0.0
        self._memory_history: deque = deque(maxlen=10)
        self._last_cleanup_time = 0.0
        self._cleanup_cooldown_seconds = 2.0

        if self._cuda_available:
            try:
                device_properties = torch.cuda.get_device_properties(0)
                self._device_name = device_properties.name
                self._total_memory_gb = device_properties.total_memory / (1024 ** 3)

                os.environ.setdefault(
                    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
                )

                logger.info(
                    f"Memory manager initialized: GPU={self._device_name}, "
                    f"total={self._total_memory_gb:.2f}GB"
                )
            except Exception as e:
                logger.error(f"CUDA init failed: {e}")
                self._cuda_available = False
        else:
            logger.warning("CUDA unavailable, memory manager running in CPU mode")

    def _update_memory_history(self, used_gb: float):
        """Append a (timestamp, used_gb) sample to the history deque."""
        self._memory_history.append((time.time(), used_gb))

    def get_memory_trend(self) -> str:
        """Classify recent VRAM usage trend.

        Compares the oldest and newest of the last 3 samples:
          - delta > 0.5 GiB → "increasing"
          - delta < -0.5 GiB → "decreasing"
          - otherwise → "stable"

        Returns:
            One of "increasing", "decreasing", "stable".
        """
        if len(self._memory_history) < 3:
            return "stable"

        recent = list(self._memory_history)[-3:]
        oldest_used = recent[0][1]
        newest_used = recent[-1][1]
        delta = newest_used - oldest_used

        if delta > 0.5:
            return "increasing"
        elif delta < -0.5:
            return "decreasing"
        return "stable"

    def get_memory_stats(self) -> MemoryStats:
        """Return a MemoryStats snapshot of current GPU VRAM usage.

        Uses torch.cuda.mem_get_info() for driver-level stats (includes all
        processes, not just the current PyTorch process).

        Returns:
            MemoryStats with current VRAM figures. If CUDA is unavailable,
            all numeric fields are 0.0 and available=False.
        """
        if not self._cuda_available:
            return MemoryStats(
                available=False,
                total_memory=0.0,
                used_memory=0.0,
                free_memory=0.0,
                utilization_percent=0.0,
                device_name="CPU",
            )

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            total_gb = total_bytes / (1024 ** 3)
            free_gb = free_bytes / (1024 ** 3)
            used_gb = total_gb - free_gb
            utilization = (used_gb / total_gb * 100) if total_gb > 0 else 0.0

            self._update_memory_history(used_gb)

            return MemoryStats(
                available=True,
                total_memory=round(total_gb, 2),
                used_memory=round(used_gb, 2),
                free_memory=round(free_gb, 2),
                utilization_percent=round(utilization, 1),
                device_name=self._device_name,
            )
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return MemoryStats(
                available=False,
                total_memory=0.0,
                used_memory=0.0,
                free_memory=0.0,
                utilization_percent=0.0,
                device_name="Error",
            )

    def check_and_cleanup(
        self,
        required_memory_gb: float = 1.0,
        cleanup_threshold_gb: float = 1.5,
    ) -> bool:
        """Check if enough VRAM is available; trigger cleanup if not.

        Args:
            required_memory_gb: Minimum free VRAM needed (GiB).
            cleanup_threshold_gb: Free VRAM below this triggers cleanup.

        Returns:
            True if enough VRAM is available (after optional cleanup),
            False if still insufficient.
        """
        if not self._cuda_available:
            logger.debug("CUDA unavailable, skipping memory check")
            return True

        stats = self.get_memory_stats()
        free_memory = stats.free_memory

        logger.debug(
            f"Memory check: total={stats.total_memory:.2f}GB, "
            f"used={stats.used_memory:.2f}GB, "
            f"free={free_memory:.2f}GB, "
            f"need={required_memory_gb:.2f}GB, "
            f"util={stats.utilization_percent:.1f}%"
        )

        if free_memory >= required_memory_gb:
            return True

        current_time = time.time()
        time_since_last_cleanup = current_time - self._last_cleanup_time

        if time_since_last_cleanup < self._cleanup_cooldown_seconds:
            logger.debug(
                f"Cleanup cooldown ({time_since_last_cleanup:.1f}s < "
                f"{self._cleanup_cooldown_seconds}s), skipping"
            )
        elif free_memory < cleanup_threshold_gb:
            trend = self.get_memory_trend()
            aggressive = (trend == "increasing") or (stats.utilization_percent > 85)

            logger.warning(
                f"Low VRAM ({free_memory:.2f}GB < {cleanup_threshold_gb}GB), "
                f"trend={trend}, triggering "
                f"{'aggressive' if aggressive else 'normal'} cleanup"
            )

            self.cleanup(aggressive=aggressive)

            stats = self.get_memory_stats()
            free_memory = stats.free_memory
            logger.info(
                f"Post-cleanup: used={stats.used_memory:.2f}GB, "
                f"free={free_memory:.2f}GB, "
                f"util={stats.utilization_percent:.1f}%"
            )

        if free_memory < required_memory_gb:
            logger.error(
                f"Insufficient VRAM: need {required_memory_gb:.2f}GB, "
                f"have {free_memory:.2f}GB"
            )
            return False

        return True

    def cleanup(self, aggressive: bool = False):
        """Release GPU VRAM.

        Normal: gc.collect() → torch.cuda.empty_cache().
        Aggressive: above + unload inactive models → second empty_cache().

        Order matters: gc.collect() must precede empty_cache() so that
        Python objects are freed first, making their VRAM available for
        the cache allocator to release back to CUDA.

        Args:
            aggressive: If True, also unload inactive models (default 5 min).
        """
        if not self._cuda_available:
            return

        self._last_cleanup_time = time.time()
        logger.info(f"Starting {'aggressive' if aggressive else 'normal'} VRAM cleanup...")

        gc.collect()
        torch.cuda.empty_cache()

        if aggressive:
            try:
                from core.model_manager import model_manager

                model_manager.unload_inactive_models(inactive_threshold_minutes=5)
                torch.cuda.empty_cache()
                logger.info("Unloaded inactive models and completed VRAM cleanup")
            except Exception as e:
                logger.warning(f"Failed to unload models: {e}")

        logger.debug("VRAM cleanup complete")

    def log_memory_usage(self, context: str = ""):
        """Log a VRAM snapshot with trend indicator.

        Args:
            context: Label for the log entry, e.g. "LLM加载前".
        """
        if not self._cuda_available:
            logger.info(f"{context} - CPU mode")
            return

        stats = self.get_memory_stats()
        trend = self.get_memory_trend()
        trend_icon = {
            "increasing": "[+]", "decreasing": "[-]", "stable": "[=]"
        }.get(trend, "[=]")

        logger.info(
            f"{context} - GPU={stats.device_name}, "
            f"used={stats.used_memory:.2f}GB, "
            f"free={stats.free_memory:.2f}GB, "
            f"total={stats.total_memory:.2f}GB, "
            f"util={stats.utilization_percent:.1f}% {trend_icon}"
        )


_lock = threading.Lock()
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(settings=None) -> MemoryManager:
    """Return the thread-safe MemoryManager singleton via double-checked locking.

    Args:
        settings: Settings object (accepted for API compatibility, not used).

    Returns:
        The global MemoryManager instance.
    """
    global _memory_manager

    if _memory_manager is None:
        with _lock:
            if _memory_manager is None:
                _memory_manager = MemoryManager()

    return _memory_manager
