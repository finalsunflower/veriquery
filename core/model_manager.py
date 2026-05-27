"""
Model manager — GPU model registry with lifecycle management.

Provides a singleton ModelManager that acts as a global registry for
all loaded deep-learning models (LLM, VLM, CLIP, etc.). Supports
registration, lookup, individual/batch unload, and LRU-based
automatic cleanup of inactive models.
"""

import logging
import time
import torch
from typing import Dict, Any, List
from threading import Lock

logger = logging.getLogger(__name__)


class ModelManager:
    """Global model registry with LRU-based lifecycle management.

    Singleton via __new__ + double-checked locking. All reads/writes to
    _models and _model_last_used are protected by _global_lock.

    Attributes:
        _models: Registry mapping model name → model object.
        _model_last_used: Registry mapping model name → last-use timestamp.
        _global_lock: Mutex protecting shared data access.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._models: Dict[str, Any] = {}
        self._model_last_used: Dict[str, float] = {}
        self._global_lock = Lock()
        self._initialized = True
        logger.info("ModelManager initialized")

    def register_model(self, name: str, model: Any):
        """Register a loaded model under the given name.

        Args:
            name: Unique identifier for the model (e.g. "qwen35", "clip").
            model: The loaded model object (any framework).
        """
        with self._global_lock:
            self._models[name] = model
            self._model_last_used[name] = time.time()
        logger.info(f"Model registered: {name}")

    def get_model(self, name: str) -> Any:
        """Retrieve a registered model by name and update its last-used timestamp.

        Args:
            name: Model identifier used during registration.

        Returns:
            The model object, or None if not found.
        """
        with self._global_lock:
            model = self._models.get(name)
            if model is not None:
                self._model_last_used[name] = time.time()
        if model is None:
            logger.warning(f"Model not found: {name}")
        return model

    def unload_model(self, name: str) -> bool:
        """Unload a single model by name, releasing its GPU VRAM.

        Deletes the model reference and calls torch.cuda.empty_cache()
        to return freed blocks to the CUDA driver.

        Args:
            name: Model identifier to unload.

        Returns:
            True if the model was found and unloaded, False otherwise.
        """
        with self._global_lock:
            if name not in self._models:
                logger.warning(f"Cannot unload '{name}': not registered")
                return False

            model = self._models.pop(name)
            self._model_last_used.pop(name, None)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Model unloaded: {name}")
        return True

    def unload_all(self):
        """Unload all registered models.

        Takes a snapshot of model names under lock, then unloads each
        individually (each unload_model call acquires the lock independently).
        """
        logger.info("Unloading all models...")

        with self._global_lock:
            model_names = list(self._models.keys())

        unloaded_count = 0
        for model_name in model_names:
            if self.unload_model(model_name):
                unloaded_count += 1

        logger.info(f"All models unloaded ({unloaded_count}/{len(model_names)})")

    def unload_inactive_models(self, inactive_threshold_minutes: int = 10) -> List[str]:
        """Unload models that have been idle longer than the given threshold.

        Uses an LRU strategy: models whose last-use timestamp is older than
        *inactive_threshold_minutes* are collected under lock, then unloaded
        one by one outside the lock to avoid nested locking.

        Args:
            inactive_threshold_minutes: Idle time threshold in minutes.
                Default 10; memory_manager passes 5 during aggressive cleanup.

        Returns:
            List of names of successfully unloaded models.
        """
        current_time = time.time()
        threshold_seconds = inactive_threshold_minutes * 60

        models_to_unload = []

        with self._global_lock:
            for model_name, last_used in self._model_last_used.items():
                if model_name not in self._models:
                    continue
                inactive_duration = current_time - last_used
                if inactive_duration > threshold_seconds:
                    models_to_unload.append(model_name)
                    logger.debug(
                        f"Model {model_name} inactive for "
                        f"{inactive_duration / 60:.1f} minutes"
                    )

        unloaded_models = []
        if models_to_unload:
            logger.info(
                f"Found {len(models_to_unload)} inactive model(s), unloading..."
            )
            for model_name in models_to_unload:
                if self.unload_model(model_name):
                    unloaded_models.append(model_name)
            logger.info(
                f"Inactive models unloaded "
                f"({len(unloaded_models)}/{len(models_to_unload)})"
            )
        else:
            logger.info("No inactive models to unload")

        return unloaded_models


model_manager = ModelManager()
