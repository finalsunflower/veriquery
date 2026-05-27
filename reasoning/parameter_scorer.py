"""
Enhanced Parameter Scoring Engine - Three-Layer Architecture

Architecture Position:
    User Request → api/routers/compare.py (API Layer)
                 → agents/comparison_node.py (Agent Orchestration Layer)
                 → ★ reasoning/parameter_scorer.py (Core Scoring Engine) ★
                 ← Returns ScoringResult (rankings, recommendations, dimension scores)

Upstream: extraction/parameter_extractor.py provides electrical parameters
Downstream: agents/comparison_node.py invokes this engine for chip comparison scoring
Peer: reasoning/erc_engine.py (Electrical Rule Check Engine)

Three-Layer Architecture:
    Layer 1: CCM (Context-Condition Mapping) - Test condition normalization
        Maps parameters measured under different temperature/voltage conditions
        to a unified standard (25°C, 3.3V) with confidence attenuation.

    Layer 2: Z-A-FoM (Z-number Augmented Figure of Merit) - Reliability fusion
        Assigns different reliability levels based on data source (datasheet,
        knowledge graph, LLM inference, etc.) and applies Z-number Kang
        conversion to naturally penalize low-reliability data in scoring.

    Layer 3: B-SPOTIS (Balanced SPOTIS) - Robust decision engine
        Min-Max normalization → MEREC objective weighting → ESP distance
        calculation → reliability fusion → final ranking.

Academic References:
    - SPOTIS: Dezert, J., et al. (2020). The SPOTIS method for multi-criteria
      decision-making problems. Information Sciences, 517, 452-469.
    - Z-numbers: Zadeh, L. A. (2011). A note on Z-numbers. Information Sciences,
      181(22), 2923-2932.
    - Kang conversion: Kang, B., et al. (2012). A method of converting Z-number
      to classical fuzzy number. J. of Information and Computational Science,
      9(3), 703-709.
    - MEREC: Keshavarz-Ghorabaee, M., et al. (2021). Determination of objective
      weights using MEREC. Symmetry, 13(4), 525.
"""

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TriangularFuzzyNumber:
    """Triangular Fuzzy Number (TFN) represented as (l, m, r).

    A TFN is the building block of the Z-number system. Each Z-number consists
    of two TFNs: A (fuzzy constraint on value) and B (fuzzy constraint on
    reliability).

    The membership function is:
        μ(x) = (x-l)/(m-l)   for l ≤ x ≤ m
        μ(x) = (r-x)/(r-m)   for m ≤ x ≤ r
        μ(x) = 0              otherwise

    Attributes:
        l: Lower bound where μ(x) = 0.
        m: Modal value (peak) where μ(x) = 1.
        r: Upper bound where μ(x) = 0.
    """

    l: float
    m: float
    r: float

    def __post_init__(self):
        if not (self.l <= self.m <= self.r):
            raise ValueError(f"TFN requires l≤m≤r: l={self.l}, m={self.m}, r={self.r}")

    @property
    def spread(self) -> float:
        """Total spread = r - l, reflecting the uncertainty range."""
        return self.r - self.l

    @property
    def left_spread(self) -> float:
        """Left spread = m - l, used in Kang conversion for proportional expansion."""
        return self.m - self.l

    @property
    def right_spread(self) -> float:
        """Right spread = r - m, used in Kang conversion for proportional expansion."""
        return self.r - self.m

    def defuzzify(self) -> float:
        """Defuzzify using Graded Mean Integration Representation.

        Formula: x* = (l + 4m + r) / 6

        The 1:4:1 weighting gives the modal value m higher weight, making the
        result closer to the most likely value. When l=m=r (crisp number),
        the result degenerates to m itself, ensuring consistency.
        """
        return (self.l + 4 * self.m + self.r) / 6

    @classmethod
    def from_crisp(cls, value: float, spread_ratio: float = 0.1) -> 'TriangularFuzzyNumber':
        """Construct a TFN from a crisp value with proportional spread.

        Args:
            value: Crisp value to be used as the modal value m.
            spread_ratio: Proportional spread (default 0.1 = ±10% uncertainty).
                - Datasheet direct read: 0.05 (±5%)
                - Graph estimation: 0.15 (±15%)
                - LLM inference: 0.25 (±25%)

        Returns:
            A TriangularFuzzyNumber centered at value with proportional spread.
        """
        spread = abs(value) * spread_ratio
        return cls(l=value - spread, m=value, r=value + spread)

    @classmethod
    def from_uncertainty(cls, value: float, uncertainty: float) -> 'TriangularFuzzyNumber':
        """Construct a TFN from a crisp value with known absolute uncertainty.

        Args:
            value: Crisp value to be used as the modal value m.
            uncertainty: Absolute uncertainty, used directly as left/right spread.
                e.g., value=100, uncertainty=5 → TFN(95, 100, 105)

        Returns:
            A TriangularFuzzyNumber centered at value with symmetric spread.
        """
        return cls(l=value - uncertainty, m=value, r=value + uncertainty)


@dataclass
class ZNumberCore:
    """Z-number core implementation: Z = (A, B).

    A Z-number (Zadeh, 2011) extends traditional fuzzy numbers by simultaneously
    considering:
        - A: Fuzzy constraint on the variable value ("what the value roughly is")
        - B: Fuzzy constraint on the reliability of A ("how reliable this estimate is")

    This implementation uses a modified Kang conversion that performs horizontal
    spread expansion instead of the original vertical scaling:

        Original Kang: μ_A'(x) = α · μ_A(x)  (vertical scaling)
        → Problem: defuzzified value unchanged, cannot penalize low reliability

        This implementation: expand left/right spread by factor = 1 + (1 - α)
        → α=1 (fully reliable): factor=1, spread unchanged, defuzzified value = m
        → α=0.5 (moderate): factor=1.5, spread ×1.5, defuzzified value deviates from m
        → α=0 (unreliable): factor=2, spread ×2, defuzzified value deviates significantly

    Attributes:
        A: TriangularFuzzyNumber for the value constraint.
        B: TriangularFuzzyNumber for the reliability constraint (must be in [0, 1]).
    """

    A: TriangularFuzzyNumber
    B: TriangularFuzzyNumber

    def __post_init__(self):
        if not (0 <= self.B.l <= self.B.m <= self.B.r <= 1):
            raise ValueError(
                f"Reliability B must be in [0,1]: B=({self.B.l}, {self.B.m}, {self.B.r})"
            )

    def convert_to_fuzzy(self) -> TriangularFuzzyNumber:
        """Convert Z-number to classical fuzzy number via reliability-weighted spread expansion.

        Algorithm:
            1. Defuzzify B to get α (scalar reliability), clipped to [0.01, 0.99]
            2. Compute expansion factor = 1 + (1 - α)
            3. Expand left/right spreads of A proportionally, anchored at m

        Example (symmetric TFN):
            A = TFN(90, 100, 110), α=0.8
            factor = 1.2, new TFN = (88, 100, 112)
            defuzzified = (88 + 400 + 112)/6 = 100 (unchanged for symmetric TFN)

        Example (asymmetric TFN):
            A = TFN(95, 100, 110), α=0.5
            factor = 1.5, new TFN = (92.5, 100, 115)
            defuzzified = (92.5 + 400 + 115)/6 = 101.25 (deviates from peak 100)

        Returns:
            A TriangularFuzzyNumber with expanded spreads reflecting reliability.
        """
        alpha = np.clip(self.B.defuzzify(), 0.01, 0.99)
        factor = 1 + (1 - alpha)
        return TriangularFuzzyNumber(
            l=self.A.m - self.A.left_spread * factor,
            m=self.A.m,
            r=self.A.m + self.A.right_spread * factor,
        )

    def weighted_value(self) -> float:
        """Get the weighted value = defuzzified value of the converted fuzzy number.

        For symmetric TFNs, the weighted value equals m regardless of reliability,
        but the expanded spread changes the normalization range in B-SPOTIS,
        indirectly affecting all normalized scores. The Z-number penalty mechanism
        thus operates through "changing the decision matrix value range" rather
        than "changing individual defuzzified values."
        """
        return self.convert_to_fuzzy().defuzzify()

    @classmethod
    def from_crisp(
        cls,
        value: float,
        reliability: float,
        value_spread: float = 0.1,
        reliability_spread: float = 0.1,
    ) -> 'ZNumberCore':
        """Construct a ZNumberCore from crisp value and reliability.

        Args:
            value: Crisp parameter value, used as A's modal value.
            reliability: Scalar reliability in [0,1], used as B's modal value.
            value_spread: Proportional spread for A (default 10%).
            reliability_spread: Proportional spread for B (default 10%).

        Returns:
            A ZNumberCore with A and B constructed from the given parameters.
        """
        A = TriangularFuzzyNumber.from_crisp(value, value_spread)
        r = np.clip(reliability, 0.01, 0.99)
        B = TriangularFuzzyNumber.from_crisp(r, reliability_spread)
        B = TriangularFuzzyNumber(l=max(0.0, B.l), m=B.m, r=min(1.0, B.r))
        return cls(A=A, B=B)

    @staticmethod
    def aggregate(z_numbers: list, weights: Optional[list] = None) -> 'ZNumberCore':
        """Weighted aggregation of multiple Z-numbers.

        Performs weighted averaging of A and B components separately. TFN's
        closure under linear operations guarantees the result is still a valid
        triangular fuzzy number.

        Args:
            z_numbers: List of ZNumberCore instances to aggregate.
            weights: Weight list (default: equal weights). Must match z_numbers length.

        Returns:
            An aggregated ZNumberCore.

        Raises:
            ValueError: If z_numbers is empty or weights length mismatches.
        """
        if not z_numbers:
            raise ValueError("Z-number list cannot be empty")
        if weights is None:
            weights = [1.0 / len(z_numbers)] * len(z_numbers)
        if len(weights) != len(z_numbers):
            raise ValueError("Weights count must match Z-number count")
        return ZNumberCore(
            A=TriangularFuzzyNumber(
                l=sum(w * z.A.l for w, z in zip(weights, z_numbers)),
                m=sum(w * z.A.m for w, z in zip(weights, z_numbers)),
                r=sum(w * z.A.r for w, z in zip(weights, z_numbers)),
            ),
            B=TriangularFuzzyNumber(
                l=sum(w * z.B.l for w, z in zip(weights, z_numbers)),
                m=sum(w * z.B.m for w, z in zip(weights, z_numbers)),
                r=sum(w * z.B.r for w, z in zip(weights, z_numbers)),
            ),
        )


class ParameterType(Enum):
    """Parameter type enumeration determining normalization and scoring direction.

    - BENEFIT: Higher is better (e.g., frequency, output current).
        Normalization: r_ij = (x_ij - min) / (max - min)
    - COST: Lower is better (e.g., power consumption, propagation delay).
        Normalization: r_ij = (max - x_ij) / (max - min)
    - TARGET: Closer to target is better (e.g., supply voltage targeting 3.3V).
        Normalization: r_ij = 1 - |x_ij - target| / max_diff
    """

    BENEFIT = "benefit"
    COST = "cost"
    TARGET = "target"


class DataSource(Enum):
    """Data source enumeration determining Layer 2 reliability assignment.

    Reliability ranking (descending):
        1. DATASHEET_DIRECT (0.95): Directly read from datasheet
        2. DATASHEET_TABLE (0.90): From datasheet tables (may have rounding)
        3. DATASHEET_GRAPH (0.75): Estimated from datasheet graphs
        4. KNOWLEDGE_GRAPH (0.65): From knowledge graph (may be stale)
        5. LLM_INFERENCE (0.50): LLM-generated (hallucination risk)
        6. DEFAULT_VALUE (0.30): Default/missing value
    """

    DATASHEET_DIRECT = "datasheet_direct"
    DATASHEET_TABLE = "datasheet_table"
    DATASHEET_GRAPH = "datasheet_graph"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    LLM_INFERENCE = "llm_inference"
    DEFAULT_VALUE = "default_value"


@dataclass
class ZNumber:
    """Z-number data structure (external interface layer).

    Provides a simplified value/reliability interface for upper-layer code while
    internally using ZNumberCore for precise fuzzy computations. This is a
    Facade pattern that hides ZNumberCore complexity.

    Attributes:
        value: Original crisp parameter value.
        reliability: Scalar reliability in [0,1], determined by data source.
        source: Data source identifier for traceability and debugging.
    """

    value: float
    reliability: float
    source: str = "unknown"
    _core: Optional[ZNumberCore] = field(default=None, repr=False)

    def __post_init__(self):
        self.reliability = np.clip(self.reliability, 0.0, 1.0)
        if self._core is None:
            self._core = ZNumberCore.from_crisp(
                value=self.value,
                reliability=self.reliability,
            )

    @property
    def fuzzy(self) -> ZNumberCore:
        """Access the internal ZNumberCore for precise fuzzy computations."""
        return self._core

    def weighted_value(self) -> float:
        """Get the Z-number weighted value = defuzzified value after Kang conversion."""
        return self._core.weighted_value()

    def uncertainty_adjusted_value(self) -> float:
        """Get uncertainty-adjusted value (equivalent to weighted_value).

        Provided as an alternative semantic name for contexts emphasizing
        uncertainty adjustment rather than reliability weighting.
        """
        return self._core.convert_to_fuzzy().defuzzify()


@dataclass
class DeviceScore:
    """Scoring result for a single device.

    Attributes:
        device_name: Device identifier (e.g., "SN74HC04").
        overall_score: Overall score (0-100).
        reliability_score: Data reliability score (0-1).
        esp_distance: Weighted Euclidean distance to ESP.
        dimension_scores: Dimension scores (performance/power/reliability/usability), each 0-100.
        parameter_scores: Per-parameter normalized scores, 0-1.
        parameter_reliabilities: Per-parameter data reliability, 0-1.
        advantages: List of advantage descriptions.
        disadvantages: List of disadvantage descriptions.
        rank: Final ranking (1 = best).
    """

    device_name: str
    overall_score: float
    reliability_score: float
    esp_distance: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    parameter_scores: Dict[str, float] = field(default_factory=dict)
    parameter_reliabilities: Dict[str, float] = field(default_factory=dict)
    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)
    rank: int = 0


@dataclass
class ScoringResult:
    """Complete scoring result data structure.

    Attributes:
        devices: List of DeviceScore sorted by ranking.
        parameter_weights: Parameter weights from MEREC objective weighting.
        objective_weights: Criterion weights (currently same as parameter_weights).
        reliability_weights: Reliability weights (currently same as parameter_weights).
        recommendation: Natural language recommendation text.
        methodology: Methodology identifier string.
    """

    devices: List[DeviceScore]
    parameter_weights: Dict[str, float]
    objective_weights: Dict[str, float]
    reliability_weights: Dict[str, float]
    recommendation: str
    methodology: str = "CCM + Z-A-FoM + B-SPOTIS"


class ContextConditionMapper:
    """Layer 1: Test Condition Normalization Module (CCM - Context-Condition Mapping).

    Eliminates unfair comparisons caused by different test conditions across
    chip datasheets. Executes before Layer 2 (Z-A-FoM) to ensure parameter
    values fed into Z-numbers are comparable.

    Normalization Method:
        Uses semiconductor physics characteristics for linear equivalent conversion:
            correction = 1 + (coeff - 1) × (ΔT / 100)
            normalized_value = value / correction

    Confidence Attenuation:
        The further the conversion distance, the lower the confidence:
        - |ΔT| > 50°C: confidence × 0.90
        - |ΔT| > 100°C: confidence × 0.85 (cumulative)
        - |ΔV| > 0.5V: confidence × 0.92
    """

    STANDARD_CONDITIONS = {"temperature": 25.0, "voltage": 3.3, "current": 1.0}

    TEMP_COEFFICIENTS = {
        "Rds_on": {"coeff": 1.4, "type": "resistance"},
        "propagation_delay": {"coeff": 1.2, "type": "delay"},
        "power_consumption": {"coeff": 1.3, "type": "power"},
        "quiescent_current": {"coeff": 1.25, "type": "current"},
        "input_current": {"coeff": 1.15, "type": "current"},
        "output_current": {"coeff": 0.95, "type": "current"},
        "frequency": {"coeff": 0.98, "type": "frequency"},
    }

    VOLTAGE_COEFFICIENTS = {
        "propagation_delay": {"coeff": 0.9, "type": "delay"},
        "power_consumption": {"coeff": 1.2, "type": "power"},
        "output_current": {"coeff": 1.1, "type": "current"},
    }

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CCM")

    def normalize_parameter(
        self, param_name: str, value: float, test_conditions: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float, bool]:
        """Normalize a parameter to standard test conditions.

        Algorithm:
            1. If no test conditions, return original value with default confidence 0.85
            2. Temperature correction: if parameter has temp coefficient and ΔT ≠ 0
               - Compute correction = 1 + (coeff-1) × (ΔT/100)
               - Normalized value = original / correction
               - Attenuate confidence based on temperature difference
            3. Voltage correction: if parameter has voltage coefficient and ΔV > 0.5V
               - Compute correction = 1 + (coeff-1) × (ΔV/V_standard)
               - Normalized value = already-corrected / correction
               - Attenuate confidence × 0.92

        Args:
            param_name: Parameter name (e.g., "Rds_on", "propagation_delay").
            value: Original parameter value measured under test_conditions.
            test_conditions: Test condition dict, e.g., {"temperature": 125, "voltage": 5.0}.

        Returns:
            Tuple of (normalized_value, confidence, was_adjusted):
                - normalized_value: Value normalized to 25°C/3.3V
                - confidence: Confidence in [0,1], lower for larger conversion distances
                - was_adjusted: Whether condition correction was applied
        """
        if test_conditions is None or len(test_conditions) == 0:
            return value, 0.85, False

        normalized_value = value
        confidence = 1.0
        was_adjusted = False

        temp_coeff_data = self.TEMP_COEFFICIENTS.get(param_name)
        if temp_coeff_data and "temperature" in test_conditions:
            actual_temp = test_conditions["temperature"]
            standard_temp = self.STANDARD_CONDITIONS["temperature"]
            temp_diff = actual_temp - standard_temp

            if abs(temp_diff) > 1:
                coeff = temp_coeff_data["coeff"]
                p_type = temp_coeff_data["type"]

                if p_type in ["resistance", "delay", "power", "current"]:
                    correction = 1 + (coeff - 1) * (temp_diff / 100)
                    if correction > 0:
                        normalized_value = value / correction
                        was_adjusted = True

                        if abs(temp_diff) > 50:
                            confidence *= 0.90
                        if abs(temp_diff) > 100:
                            confidence *= 0.85

        volt_coeff_data = self.VOLTAGE_COEFFICIENTS.get(param_name)
        if volt_coeff_data and "voltage" in test_conditions:
            actual_volt = test_conditions["voltage"]
            standard_volt = self.STANDARD_CONDITIONS["voltage"]
            volt_diff = actual_volt - standard_volt

            if abs(volt_diff) > 0.5:
                coeff = volt_coeff_data["coeff"]
                correction = 1 + (coeff - 1) * (volt_diff / standard_volt)
                if correction > 0:
                    normalized_value = normalized_value / correction
                    was_adjusted = True
                    confidence *= 0.92

        return normalized_value, confidence, was_adjusted


class ZAugmentedFoM:
    """Layer 2: Z-number Augmented Figure of Merit Calculator (Z-A-FoM).

    Fuses parameter values and data source reliability into Z-numbers.
    Receives Layer 1 (CCM) normalized values and confidence, combines with
    data source type to construct ZNumber objects for Layer 3 (B-SPOTIS).

    Core Mechanism:
        Different data sources → different base reliability → different Z-number
        spread → natural penalty for low-reliability data in scoring.
    """

    SOURCE_RELIABILITY = {
        DataSource.DATASHEET_DIRECT: 0.95,
        DataSource.DATASHEET_TABLE: 0.90,
        DataSource.DATASHEET_GRAPH: 0.75,
        DataSource.KNOWLEDGE_GRAPH: 0.65,
        DataSource.LLM_INFERENCE: 0.50,
        DataSource.DEFAULT_VALUE: 0.30,
    }

    VALUE_SPREAD_MAPPING = {
        DataSource.DATASHEET_DIRECT: 0.05,
        DataSource.DATASHEET_TABLE: 0.08,
        DataSource.DATASHEET_GRAPH: 0.15,
        DataSource.KNOWLEDGE_GRAPH: 0.20,
        DataSource.LLM_INFERENCE: 0.25,
        DataSource.DEFAULT_VALUE: 0.30,
    }

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ZAFoM")

    def create_z_number(
        self,
        param_name: str,
        value: float,
        source: DataSource = DataSource.DATASHEET_TABLE,
        test_condition_confidence: float = 1.0,
        is_interpolated: bool = False,
    ) -> ZNumber:
        """Create a Z-number by fusing data source reliability and CCM confidence.

        Algorithm:
            1. Get base reliability and spread from data source mapping
            2. final_reliability = base_reliability × CCM_confidence (dual attenuation)
            3. If condition-adjusted (is_interpolated), further attenuate:
               - reliability × 0.90, spread × 1.2
            4. Construct ZNumberCore and wrap as ZNumber

        Args:
            param_name: Parameter name (e.g., "frequency").
            value: Parameter value (already normalized by CCM).
            source: Data source type.
            test_condition_confidence: Confidence output from Layer 1 (CCM).
            is_interpolated: Whether the value was condition-adjusted by CCM.

        Returns:
            A ZNumber incorporating source reliability and CCM confidence.
        """
        base_reliability = self.SOURCE_RELIABILITY.get(source, 0.5)
        value_spread = self.VALUE_SPREAD_MAPPING.get(source, 0.15)

        base_reliability *= test_condition_confidence

        if is_interpolated:
            base_reliability *= 0.90
            value_spread *= 1.2

        core = ZNumberCore.from_crisp(
            value=value,
            reliability=base_reliability,
            value_spread=value_spread,
            reliability_spread=0.10,
        )

        return ZNumber(
            value=value,
            reliability=base_reliability,
            source=source.value,
            _core=core,
        )

    def create_z_number_with_uncertainty(
        self,
        param_name: str,
        value: float,
        uncertainty: float,
        source: DataSource = DataSource.DATASHEET_TABLE,
        test_condition_confidence: float = 1.0,
        is_interpolated: bool = False,
    ) -> ZNumber:
        """Create a Z-number with known absolute uncertainty.

        Used when datasheets provide "typical value ± error" format.

        Args:
            param_name: Parameter name.
            value: Parameter value.
            uncertainty: Absolute uncertainty (e.g., ±5 for value=100).
            source: Data source type.
            test_condition_confidence: CCM confidence.
            is_interpolated: Whether the value was condition-adjusted.

        Returns:
            A ZNumber with uncertainty-based spread for the A component.
        """
        base_reliability = self.SOURCE_RELIABILITY.get(source, 0.5)
        base_reliability *= test_condition_confidence

        if is_interpolated:
            base_reliability *= 0.90

        A = TriangularFuzzyNumber.from_uncertainty(value, uncertainty)
        B = TriangularFuzzyNumber.from_crisp(base_reliability, 0.10)
        B = TriangularFuzzyNumber(l=max(0.0, B.l), m=B.m, r=min(1.0, B.r))
        core = ZNumberCore(A=A, B=B)

        return ZNumber(
            value=value,
            reliability=base_reliability,
            source=source.value,
            _core=core,
        )

    def aggregate_z_numbers(
        self,
        z_numbers: List[ZNumber],
        weights: Optional[List[float]] = None,
    ) -> ZNumber:
        """Aggregate multiple Z-numbers into a single Z-number.

        Used when the same parameter has multiple sources (e.g., datasheet +
        knowledge graph). Performs weighted averaging of A and B components.

        Args:
            z_numbers: List of ZNumber instances to aggregate.
            weights: Weight list (default: equal weights).

        Returns:
            An aggregated ZNumber.

        Raises:
            ValueError: If z_numbers is empty.
        """
        if not z_numbers:
            raise ValueError("Z-number list cannot be empty")

        fuzzy_numbers = [z.fuzzy for z in z_numbers]
        aggregated = ZNumberCore.aggregate(fuzzy_numbers, weights)

        aggregated_value = aggregated.weighted_value()
        aggregated_reliability = aggregated.B.defuzzify()

        return ZNumber(
            value=aggregated_value,
            reliability=aggregated_reliability,
            source="aggregated",
            _core=aggregated,
        )


class BalancedSPOTIS:
    """Layer 3: Balanced Stable Preference Ordering Engine (B-SPOTIS).

    B-SPOTIS = SPOTIS + three improvements over the original method:

    Improvement 1: Min-Max normalization replaces vector normalization
        → Solves ranking reversal problem
        → Min-Max depends only on per-column extrema, adding/removing
          alternatives does not affect existing normalized values

    Improvement 2: MEREC objective weighting replaces AHP subjective weighting
        → Solves weight subjectivity problem
        → MEREC measures criterion importance by "performance change after
          removing that criterion"

    Improvement 3: ESP (Expected Solution Point) reference point
        → Matches engineers' practical selection expectations rather than
          pursuing theoretical optimality

    Scoring Pipeline:
        1. Build decision matrix (n_devices × n_params)
        2. Min-Max normalization
        3. MEREC objective weighting
        4. Calculate ESP distance
        5. Fuse Z-number reliability
        6. Final score = base_scores × 0.7 + z_confidence × 0.3
        7. Ranking + dimension scores + advantage/disadvantage analysis
    """

    DEFAULT_ESP = {
        "frequency": 0.6,
        "propagation_delay": 0.6,
        "power_consumption": 0.6,
        "output_current": 0.6,
        "temperature_range": 0.6,
        "supply_voltage": 0.6,
        "package_size": 0.6,
        "ttl_compatible": 0.6,
        "input_current": 0.6,
        "quiescent_current": 0.6,
        "input_voltage_high": 0.6,
        "input_voltage_low": 0.6,
    }

    PARAM_DEFINITIONS = {
        "frequency": {"type": ParameterType.BENEFIT, "min": 0, "max": 1000},
        "propagation_delay": {"type": ParameterType.COST, "min": 0, "max": 100},
        "supply_voltage": {"type": ParameterType.TARGET, "min": 0, "max": 15, "target": 3.3},
        "output_current": {"type": ParameterType.BENEFIT, "min": 0, "max": 100},
        "input_voltage_high": {"type": ParameterType.COST, "min": 0, "max": 5},
        "input_voltage_low": {"type": ParameterType.BENEFIT, "min": 0, "max": 2},
        "power_consumption": {"type": ParameterType.COST, "min": 0, "max": 1000},
        "temperature_range": {"type": ParameterType.BENEFIT, "min": 0, "max": 200},
        "package_size": {"type": ParameterType.COST, "min": 0, "max": 100},
        "ttl_compatible": {"type": ParameterType.BENEFIT, "min": 0, "max": 1},
        "input_current": {"type": ParameterType.COST, "min": 0, "max": 100},
        "quiescent_current": {"type": ParameterType.COST, "min": 0, "max": 500},
    }

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.BSPOTIS")
        self._merec_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_max_size = 100

    def _generate_cache_key(self, matrix: np.ndarray, param_names: List[str]) -> str:
        """Generate a cache key for MEREC weights using MD5 hashing.

        Uses MD5 hash of the decision matrix bytes + sorted parameter names
        to ensure cache reuse for identical inputs.

        Args:
            matrix: Decision matrix as numpy array.
            param_names: List of parameter names.

        Returns:
            A string cache key combining matrix and parameter name hashes.
        """
        matrix_hash = hashlib.md5(matrix.tobytes()).hexdigest()[:16]
        params_str = ",".join(sorted(param_names))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:16]
        return f"{matrix_hash}_{params_hash}"

    def _cache_merec_weights(self, cache_key: str, weights: np.ndarray):
        """Cache MEREC weights using LRU eviction strategy.

        When the cache exceeds _cache_max_size, the least recently used entry
        is evicted.

        Args:
            cache_key: Cache key string.
            weights: MEREC weight vector to cache.
        """
        if cache_key in self._merec_cache:
            self._merec_cache.move_to_end(cache_key)
            return

        if len(self._merec_cache) >= self._cache_max_size:
            self._merec_cache.popitem(last=False)

        self._merec_cache[cache_key] = weights.copy()

    def clear_cache(self):
        """Clear the MEREC weight cache."""
        self._merec_cache.clear()
        self.logger.debug("MEREC weight cache cleared")

    def minmax_normalize(self, matrix: np.ndarray, param_names: List[str]) -> np.ndarray:
        """Min-Max normalization to avoid ranking reversal.

        Unlike vector normalization (used in TOPSIS), Min-Max normalization
        depends only on per-column extrema. While adding/removing alternatives
        may change extrema, the ESP framework uses fixed reference points,
        making ranking stability significantly better than TOPSIS.

        Normalization formulas by parameter type:
            BENEFIT: r_ij = (x_ij - min_j) / (max_j - min_j)
            COST:    r_ij = (max_j - x_ij) / (max_j - min_j)
            TARGET:  r_ij = 1 - |x_ij - target| / max_diff

        When max_j - min_j < 1e-10 (all values identical), normalized value
        is set to 0.5 (neutral) to avoid division by zero.

        Args:
            matrix: Raw decision matrix (n_devices × n_params).
            param_names: List of parameter names corresponding to columns.

        Returns:
            Normalized decision matrix with values in [0, 1].
        """
        normalized = np.zeros_like(matrix, dtype=float)
        n_params = matrix.shape[1]

        for j in range(n_params):
            col = matrix[:, j]
            param_name = param_names[j] if j < len(param_names) else f"param_{j}"
            param_def = self.PARAM_DEFINITIONS.get(param_name, {"type": ParameterType.BENEFIT})
            param_type = param_def.get("type", ParameterType.BENEFIT)

            min_val = col.min()
            max_val = col.max()

            if max_val - min_val < 1e-10:
                normalized[:, j] = 0.5
            elif param_type == ParameterType.COST:
                normalized[:, j] = (max_val - col) / (max_val - min_val)
            elif param_type == ParameterType.TARGET:
                target = param_def.get("target", (min_val + max_val) / 2)
                diff = np.abs(col - target)
                max_diff = max(np.abs(max_val - target), np.abs(min_val - target))
                if max_diff > 0:
                    normalized[:, j] = 1.0 - (diff / max_diff)
                else:
                    normalized[:, j] = 1.0
            else:
                normalized[:, j] = (col - min_val) / (max_val - min_val)

        return normalized

    def merec_weights(self, normalized_matrix: np.ndarray, param_names: List[str]) -> np.ndarray:
        """MEREC objective weighting method.

        Reference: Keshavarz-Ghorabaee, M., et al. (2021). Determination of
        objective weights using a method based on the removal effects of
        criteria (MEREC). Symmetry, 13(4), 525.

        Core Idea:
            If removing a criterion causes a large change in overall performance,
            that criterion has high importance and should receive high weight.

        Algorithm (5 steps):
            Step 1: MEREC normalization
                - BENEFIT: r_ij = x_ij / max(x_j)
                - COST: r_ij = min(x_j) / x_ij
            Step 2: Overall performance S_i = ln(1 + (1/m) × Σ|r_ij|)
            Step 3: Performance without criterion j: S'_ij
            Step 4: Removal effect E_j = Σ_i |S'_ij - S_i|
            Step 5: Normalized weights w_j = E_j / ΣE_j

        Args:
            normalized_matrix: Min-Max normalized decision matrix.
            param_names: List of parameter names.

        Returns:
            Weight vector of shape (n_params,).
        """
        n, m = normalized_matrix.shape

        if n < 2 or m < 1:
            return np.ones(m) / m if m > 0 else np.array([])

        cache_key = self._generate_cache_key(normalized_matrix, param_names)
        if cache_key in self._merec_cache:
            self._merec_cache.move_to_end(cache_key)
            self.logger.debug(f"MEREC weight cache hit: {cache_key[:16]}...")
            return self._merec_cache[cache_key]

        r = np.zeros_like(normalized_matrix, dtype=float)

        for j in range(m):
            col = normalized_matrix[:, j]
            param_name = param_names[j] if j < len(param_names) else f"param_{j}"
            param_def = self.PARAM_DEFINITIONS.get(param_name, {"type": ParameterType.BENEFIT})
            param_type = param_def.get("type", ParameterType.BENEFIT)

            col_min = col.min()
            col_max = col.max()

            if col_max - col_min < 1e-10:
                r[:, j] = 1.0
            elif param_type == ParameterType.COST:
                r[:, j] = col_min / np.where(col > 0, col, 1e-10)
            else:
                r[:, j] = col / np.where(col_max > 0, col_max, 1)

        S = np.zeros(n)
        for i in range(n):
            S[i] = np.log(1 + (1.0 / m) * np.sum(np.abs(r[i, :])))

        E = np.zeros(m)
        for j in range(m):
            removal_effect = 0.0
            for i in range(n):
                r_ij_removed = np.delete(r[i, :], j)
                m_removed = m - 1 if m > 1 else 1
                S_ij_removed = np.log(1 + (1.0 / m_removed) * np.sum(np.abs(r_ij_removed)))
                removal_effect += np.abs(S_ij_removed - S[i])
            E[j] = removal_effect

        E_sum = E.sum()
        if E_sum > 1e-10:
            W = E / E_sum
        else:
            W = np.ones(m) / m

        self._cache_merec_weights(cache_key, W)

        return W

    def calculate_esp_distance(
        self,
        normalized_matrix: np.ndarray,
        esp: Dict[str, float],
        weights: np.ndarray,
        param_names: List[str],
    ) -> np.ndarray:
        """Calculate weighted Euclidean distance to ESP (Expected Solution Point).

        Formula: D_ESP(i) = √(Σ_j w_j × (r_ij - ESP_j)²)

        Unlike TOPSIS which uses ideal/negative-ideal solutions that change
        with the alternative set, SPOTIS uses fixed ESP reference points,
        ensuring ranking stability.

        Args:
            normalized_matrix: Normalized decision matrix (n_devices × n_params).
            esp: ESP reference point dict {param_name: esp_value}.
            weights: MEREC weight vector.
            param_names: Parameter name list.

        Returns:
            Distance array of shape (n_devices,).
        """
        n_params = normalized_matrix.shape[1]

        esp_vector = np.zeros(n_params)
        for j, param_name in enumerate(param_names):
            esp_vector[j] = esp.get(param_name, 0.6)

        weighted_sq_diff = ((normalized_matrix - esp_vector) ** 2) * weights
        distances = np.sqrt(weighted_sq_diff.sum(axis=1))

        return distances

    def score_devices(
        self,
        devices_data: Dict[str, Dict[str, ZNumber]],
        user_esp: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[DeviceScore], Dict[str, float], Dict[str, float]]:
        """B-SPOTIS main scoring function.

        Complete 7-step scoring pipeline:
            Step 1: Build decision matrix and reliability matrix
            Step 2: Min-Max normalization (avoid ranking reversal)
            Step 3: MEREC objective weighting (avoid subjectivity)
            Step 4: Calculate ESP distance (match engineering expectations)
            Step 5: Fuse Z-number reliability (penalize low-confidence data)
            Step 6: Final score = base_scores × 0.7 + z_confidence × 0.3
            Step 7: Ranking + dimension scores + advantage/disadvantage analysis

        Args:
            devices_data: Device data dict {device_name: {param_name: ZNumber}}.
            user_esp: User-defined ESP preferences (optional).

        Returns:
            Tuple of (device_scores, merec_weights_dict, reliability_weights_dict).

        Raises:
            ValueError: If fewer than 2 devices or no parameters provided.
        """
        device_names = list(devices_data.keys())
        if len(device_names) < 2:
            raise ValueError("At least 2 devices are required for comparison")

        first_device = list(devices_data.values())[0]
        param_names = list(first_device.keys())

        if not param_names:
            raise ValueError("No parameters available for comparison")

        n_devices = len(device_names)
        n_params = len(param_names)

        value_matrix = np.zeros((n_devices, n_params))
        reliability_matrix = np.zeros((n_devices, n_params))
        missing_mask = np.zeros((n_devices, n_params), dtype=bool)

        for i, device in enumerate(device_names):
            for j, param in enumerate(param_names):
                z_num = devices_data[device].get(param)
                if z_num:
                    value_matrix[i, j] = z_num.weighted_value()
                    reliability_matrix[i, j] = z_num.reliability
                else:
                    value_matrix[i, j] = 0.0
                    reliability_matrix[i, j] = 0.3
                    missing_mask[i, j] = True

        normalized_matrix = self.minmax_normalize(value_matrix, param_names)
        normalized_matrix[missing_mask] = 0.0

        merec_w = self.merec_weights(normalized_matrix, param_names)

        esp = {p: self.DEFAULT_ESP.get(p, 0.6) for p in param_names}
        if user_esp:
            esp.update(user_esp)

        esp_distances = self.calculate_esp_distance(normalized_matrix, esp, merec_w, param_names)

        z_confidence = np.zeros(n_devices)
        for i, device in enumerate(device_names):
            for j, param in enumerate(param_names):
                z_num = devices_data[device].get(param)
                if z_num:
                    converted = z_num.fuzzy.convert_to_fuzzy()
                    if abs(converted.m) > 1e-10:
                        relative_uncertainty = converted.spread / (2.0 * abs(converted.m))
                    else:
                        relative_uncertainty = 0.0 if converted.spread < 1e-10 else 1.0
                    relative_uncertainty = min(relative_uncertainty, 1.0)
                    z_confidence[i] += merec_w[j] * (1.0 - relative_uncertainty)
                else:
                    z_confidence[i] += merec_w[j] * 0.3

        max_dist = esp_distances.max() + 1e-10
        base_scores = 1.0 - (esp_distances / max_dist)
        final_scores = base_scores * 0.7 + z_confidence * 0.3

        rankings = np.argsort(-final_scores)

        param_types = {
            p: self.PARAM_DEFINITIONS.get(p, {}).get("type", ParameterType.BENEFIT)
            for p in param_names
        }

        device_scores = []
        for rank, idx in enumerate(rankings):
            device_name = device_names[idx]

            dimension_scores = self._calculate_dimension_scores(
                normalized_matrix[idx], param_names, merec_w
            )

            param_scores = {param_names[j]: normalized_matrix[idx, j] for j in range(n_params)}
            param_reliabilities = {
                param_names[j]: reliability_matrix[idx, j] for j in range(n_params)
            }

            advantages, disadvantages = self._analyze_advantages_disadvantages(
                device_names, normalized_matrix, idx, param_names, param_types
            )

            device_scores.append(
                DeviceScore(
                    device_name=device_name,
                    overall_score=final_scores[idx] * 100,
                    reliability_score=z_confidence[idx],
                    esp_distance=esp_distances[idx],
                    dimension_scores=dimension_scores,
                    parameter_scores=param_scores,
                    parameter_reliabilities=param_reliabilities,
                    advantages=advantages,
                    disadvantages=disadvantages,
                    rank=rank + 1,
                )
            )

        merec_weights_dict = {param_names[j]: merec_w[j] for j in range(n_params)}

        return device_scores, merec_weights_dict, merec_weights_dict.copy()

    def _calculate_dimension_scores(
        self, normalized_row: np.ndarray, param_names: List[str], weights: np.ndarray
    ) -> Dict[str, float]:
        """Calculate dimension scores by grouping parameters into 4 engineering dimensions.

        Dimensions:
            - performance: frequency, propagation_delay, output_current
            - power: power_consumption, quiescent_current, input_current
            - reliability: temperature_range, ttl_compatible
            - usability: package_size, supply_voltage

        Each dimension score = weighted average of its parameters × 100.

        Args:
            normalized_row: Normalized scores for a single device.
            param_names: Parameter name list.
            weights: MEREC weight vector.

        Returns:
            Dict mapping dimension name to score (0-100).
        """
        dimensions = {
            "performance": ["frequency", "propagation_delay", "output_current"],
            "power": ["power_consumption", "quiescent_current", "input_current"],
            "reliability": ["temperature_range", "ttl_compatible"],
            "usability": ["package_size", "supply_voltage"],
        }

        dimension_scores = {}
        for dim_name, dim_params in dimensions.items():
            dim_indices = [i for i, p in enumerate(param_names) if p in dim_params]
            if dim_indices:
                dim_weights = weights[dim_indices]
                dim_values = normalized_row[dim_indices]

                if dim_weights.sum() > 0:
                    weighted_avg = (dim_values * dim_weights).sum() / dim_weights.sum()
                else:
                    weighted_avg = dim_values.mean()

                score = weighted_avg * 100
                dimension_scores[dim_name] = np.clip(score, 0.0, 100.0)
            else:
                dimension_scores[dim_name] = 50.0

        return dimension_scores

    def _analyze_advantages_disadvantages(
        self,
        all_devices: List[str],
        normalized_matrix: np.ndarray,
        device_idx: int,
        param_names: List[str],
        param_types: Dict[str, ParameterType],
    ) -> Tuple[List[str], List[str]]:
        """Analyze device advantages and disadvantages by comparing with other devices.

        Judgment rules (with 0.05 tolerance):
            - BENEFIT: score ≥ other_max - 0.05 → advantage ("excellent")
                       score ≤ other_min + 0.05 → disadvantage ("relatively weak")
            - COST: score ≥ other_max - 0.05 → advantage ("well controlled")
                    score ≤ other_min + 0.05 → disadvantage ("high")

        Results are limited to 4 advantages and 3 disadvantages to avoid
        information overload.

        Args:
            all_devices: All device name list.
            normalized_matrix: Normalized decision matrix.
            device_idx: Index of the device being analyzed.
            param_names: Parameter name list.
            param_types: Parameter type dict.

        Returns:
            Tuple of (advantages_list, disadvantages_list).
        """
        advantages = []
        disadvantages = []

        device_scores = normalized_matrix[device_idx]
        other_scores = np.delete(normalized_matrix, device_idx, axis=0)

        for j, param in enumerate(param_names):
            param_type = param_types.get(param, ParameterType.BENEFIT)
            device_score = device_scores[j]

            if len(other_scores) > 0:
                other_max = other_scores[:, j].max()
                other_min = other_scores[:, j].min()

                if param_type == ParameterType.BENEFIT:
                    if device_score >= other_max - 0.05:
                        advantages.append(f"{param}表现优秀")
                    elif device_score <= other_min + 0.05:
                        disadvantages.append(f"{param}相对较弱")
                elif param_type == ParameterType.COST:
                    if device_score >= other_max - 0.05:
                        advantages.append(f"{param}控制良好")
                    elif device_score <= other_min + 0.05:
                        disadvantages.append(f"{param}偏高")

        return advantages[:4], disadvantages[:3]


class EnhancedParameterScoringEngine:
    """Enhanced Parameter Scoring Engine - Three-Layer Architecture Integrator.

    Facade pattern: provides a unified score_devices() entry point for
    upper-layer code, hiding the complexity of the three-layer interaction.

    Layer Orchestration:
        Layer 1: CCM (self.ccm) - Test condition normalization
        Layer 2: Z-A-FoM (self.z_fom) - Z-number reliability fusion
        Layer 3: B-SPOTIS (self.b_spotis) - Robust decision engine
    """

    def __init__(self):
        self.ccm = ContextConditionMapper()
        self.z_fom = ZAugmentedFoM()
        self.b_spotis = BalancedSPOTIS()
        self.logger = logging.getLogger(f"{__name__}.EnhancedEngine")

    def score_devices(
        self,
        devices_data: Dict[str, Dict[str, Any]],
        user_preferences: Optional[Dict[str, float]] = None,
    ) -> ScoringResult:
        """Complete three-layer architecture scoring pipeline.

        Data Flow:
            Raw params → [CCM normalization] → normalized values + confidence
                       → [Z-A-FoM fusion] → ZNumber (weighted value + reliability)
                       → [B-SPOTIS decision] → DeviceScore (ranking, scores, analysis)

        Input Format:
            Simple: {device: {param: float_value}}
                → Automatically uses DATASHEET_TABLE source, no test conditions
            Detailed: {device: {param: {"value": float, "source": str, "test_conditions": dict}}}
                → Specifies data source and test conditions

        Args:
            devices_data: Device data dict with parameter values.
            user_preferences: User preference dict (optional ESP), e.g., {"frequency": 0.8}.

        Returns:
            ScoringResult containing rankings, weights, and recommendation.
        """
        self.logger.info(f"Starting three-layer architecture scoring")
        self.logger.info(f"Device list: {list(devices_data.keys())}")

        z_devices_data = {}

        for device_name, params in devices_data.items():
            z_devices_data[device_name] = {}

            for param_name, param_value in params.items():
                if isinstance(param_value, dict):
                    value = param_value.get("value", 0)
                    source_str = param_value.get("source", "datasheet_table")
                    test_conditions = param_value.get("test_conditions", {})

                    try:
                        source = DataSource(source_str)
                    except ValueError:
                        source = DataSource.DATASHEET_TABLE
                else:
                    value = param_value
                    source = DataSource.DATASHEET_TABLE
                    test_conditions = {}

                normalized_value, condition_confidence, was_adjusted = self.ccm.normalize_parameter(
                    param_name, float(value) if value else 0, test_conditions
                )

                z_num = self.z_fom.create_z_number(
                    param_name=param_name,
                    value=normalized_value,
                    source=source,
                    test_condition_confidence=condition_confidence,
                    is_interpolated=was_adjusted,
                )

                z_devices_data[device_name][param_name] = z_num

        device_scores, merec_weights, _ = (
            self.b_spotis.score_devices(z_devices_data, user_preferences)
        )

        recommendation = self._generate_recommendation(device_scores)

        self.logger.info(
            f"Scoring complete - Recommended device: "
            f"{device_scores[0].device_name if device_scores else 'N/A'}"
        )

        return ScoringResult(
            devices=device_scores,
            parameter_weights=merec_weights,
            objective_weights=merec_weights,
            reliability_weights=merec_weights,
            recommendation=recommendation,
            methodology="CCM + Z-A-FoM + B-SPOTIS",
        )

    def _generate_recommendation(
        self, device_scores: List[DeviceScore]
    ) -> str:
        """Generate natural language recommendation.

        Logic:
            1. Select the top-ranked device as recommendation
            2. Describe its overall score and reliability level
            3. List main advantages (up to 3)
            4. Mention alternative (2nd-ranked device)

        Reliability levels:
            - High (>0.8): Reliable data sources, scoring results trustworthy
            - Medium (0.6-0.8): Average data sources, results have reference value
            - Low (<0.6): Unreliable data sources, results should be used cautiously

        Args:
            device_scores: List of DeviceScore sorted by ranking.

        Returns:
            Natural language recommendation string.
        """
        if not device_scores:
            return "无法生成推荐"

        best = device_scores[0]

        reliability_desc = (
            "高"
            if best.reliability_score > 0.8
            else "中等" if best.reliability_score > 0.6 else "较低"
        )

        lines = [
            f"基于三层架构评分体系（CCM + Z-A-FoM + B-SPOTIS），",
            f"推荐选择 **{best.device_name}**，综合得分 {best.overall_score:.1f} 分，数据可靠度 {reliability_desc}。",
        ]

        if best.advantages:
            lines.append(f"主要优势：{', '.join(best.advantages[:3])}。")

        if len(device_scores) > 1:
            second = device_scores[1]
            lines.append(f"备选方案：{second.device_name}（{second.overall_score:.1f}分）。")

        return "".join(lines)


ParameterScoringEngine = EnhancedParameterScoringEngine
