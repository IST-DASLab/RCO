"""Per-layer multi-bitwidth weight cache."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class LoadMode(Enum):
    LAZY = "lazy"
    EAGER = "eager"


@dataclass
class LayerData:
    """Weight tensor and metadata for a layer at a specific bitwidth."""
    weight: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    loaded: bool = False


class WeightStore:
    """Per-layer multi-bitwidth weight cache; LAZY scans, EAGER loads upfront."""

    def __init__(self, layer_dir: str, mode: LoadMode = LoadMode.LAZY, cache: bool = False):
        self.layer_dir = Path(layer_dir)
        self.mode = mode
        self.cache = cache
        self._index: Dict[str, Dict[int, LayerData]] = {}
        self._initialized = False

    def load(self) -> "WeightStore":
        """Initialize the weight store. Returns self for chaining."""
        if self._initialized:
            logger.info("WeightStore already initialized, skipping.")
            return self

        if self.mode == LoadMode.EAGER and self.cache:
            logger.info(f"WeightStore initialized in EAGER mode from {self.layer_dir}")
            self._load_all_weights()
        else:
            logger.info(f"WeightStore initialized in LAZY mode from {self.layer_dir} (cache={self.cache})")
            self._scan_directory()

        self._initialized = True
        return self

    def _scan_directory(self) -> None:
        """Scan directory structure to build index without loading weights."""
        layer_count = 0
        weight_count = 0

        for layer_path in self.layer_dir.iterdir():
            if not layer_path.is_dir():
                continue

            layer_name = layer_path.name
            self._index[layer_name] = {}
            layer_count += 1

            for file_path in layer_path.iterdir():
                if file_path.suffix != '.pth':
                    continue

                try:
                    bitwidth = int(file_path.stem)
                except ValueError:
                    continue

                self._index[layer_name][bitwidth] = LayerData(loaded=False)
                weight_count += 1

        logger.info(f"Indexed {weight_count} weight files across {layer_count} layers")

    def _load_all_weights(self) -> None:
        """Load all weights and metadata from disk."""
        layer_count = 0
        weight_count = 0

        for layer_path in self.layer_dir.iterdir():
            if not layer_path.is_dir():
                continue

            layer_name = layer_path.name
            self._index[layer_name] = {}
            layer_count += 1

            for file_path in layer_path.iterdir():
                if file_path.suffix != '.pth':
                    continue

                try:
                    bitwidth = int(file_path.stem)
                except ValueError:
                    continue

                weight, embedded_metadata = self._load_pth(file_path)
                file_metadata = self._load_metadata_file(layer_path, bitwidth)

                metadata = {**(embedded_metadata or {}), **(file_metadata or {})} or None

                self._index[layer_name][bitwidth] = LayerData(
                    weight=weight,
                    metadata=metadata,
                    loaded=True
                )
                weight_count += 1

        logger.info(f"Loaded {weight_count} weight tensors across {layer_count} layers")

    def _load_single_weight(self, layer_name: str, bitwidth: int) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Load a single weight file from disk."""
        layer_path = self.layer_dir / layer_name
        file_path = layer_path / f"{bitwidth}.pth"

        if not file_path.exists():
            raise FileNotFoundError(f"Weight file not found: {file_path}")

        logger.debug(f"Loading weight: {layer_name} @ {bitwidth}-bit")

        weight, embedded_metadata = self._load_pth(file_path)
        file_metadata = self._load_metadata_file(layer_path, bitwidth)

        metadata = {**(embedded_metadata or {}), **(file_metadata or {})} or None

        return weight, metadata

    def _load_pth(self, path: Path) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Load a .pth file, returning (weight, metadata)."""
        data = torch.load(path, map_location='cpu', weights_only=False)

        if isinstance(data, torch.Tensor):
            return data, None

        if isinstance(data, dict):
            weight = data.get('weight')
            if weight is None:
                raise ValueError(f"No 'weight' key found in {path}")

            metadata = {k: v for k, v in data.items() if k != 'weight'}
            return weight, metadata if metadata else None

        raise ValueError(f"Unexpected data type in {path}: {type(data)}")

    def _load_metadata_file(self, layer_path: Path, bitwidth: int) -> Optional[Dict[str, Any]]:
        """Load separate metadata JSON file if it exists."""
        candidates = [
            layer_path / f"{bitwidth}-metadata.json",
            layer_path / f"{bitwidth}.json",
        ]

        for meta_path in candidates:
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    return json.load(f)

        return None

    def _validate_access(self, layer_name: str, bitwidth: int) -> None:
        """Validate that layer and bitwidth exist."""
        if not self._initialized:
            raise RuntimeError("WeightStore not initialized. Call load() first.")

        if layer_name not in self._index:
            raise KeyError(f"Layer not found: {layer_name}")

        if bitwidth not in self._index[layer_name]:
            raise KeyError(f"Bitwidth {bitwidth} not found for layer {layer_name}")

    def get_layer_weight(self, layer_name: str, bitwidth: int) -> torch.Tensor:
        """Get weight tensor for a layer at a specific bitwidth."""
        self._validate_access(layer_name, bitwidth)

        layer_data = self._index[layer_name][bitwidth]

        if self.cache and layer_data.loaded:
            return layer_data.weight

        weight, metadata = self._load_single_weight(layer_name, bitwidth)

        if self.cache:
            layer_data.weight = weight
            layer_data.metadata = metadata
            layer_data.loaded = True

        return weight

    def get_layer_metadata(self, layer_name: str, bitwidth: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a layer at a specific bitwidth."""
        self._validate_access(layer_name, bitwidth)

        layer_data = self._index[layer_name][bitwidth]

        if self.cache and layer_data.loaded:
            return layer_data.metadata

        _, metadata = self._load_single_weight(layer_name, bitwidth)

        if self.cache:
            layer_data.metadata = metadata

        return metadata

    def get_layer_names(self) -> List[str]:
        """Return list of all available layer names."""
        return list(self._index.keys())

    def get_available_bitwidths(self, layer_name: str) -> List[int]:
        """Return list of available bitwidths for a layer."""
        if layer_name not in self._index:
            raise KeyError(f"Layer not found: {layer_name}")
        return sorted(self._index[layer_name].keys())

    def get_common_bitwidths(self) -> List[int]:
        """Return bitwidths available across all layers."""
        if not self._index:
            return []

        common = set(next(iter(self._index.values())).keys())
        for layer_data in self._index.values():
            common &= set(layer_data.keys())

        return sorted(common)

    def preload_bitwidth(self, bitwidth: int) -> None:
        """Preload a specific bitwidth across all layers. Only works if cache=True."""
        if not self.cache:
            logger.warning("preload_bitwidth() called but cache=False, skipping")
            return

        for layer_name in self._index:
            if bitwidth in self._index[layer_name]:
                if not self._index[layer_name][bitwidth].loaded:
                    weight, metadata = self._load_single_weight(layer_name, bitwidth)
                    self._index[layer_name][bitwidth].weight = weight
                    self._index[layer_name][bitwidth].metadata = metadata
                    self._index[layer_name][bitwidth].loaded = True

    def clear(self) -> None:
        """Clear the cache and free memory."""
        self._index.clear()
        self._initialized = False
        logger.info("WeightStore cleared")