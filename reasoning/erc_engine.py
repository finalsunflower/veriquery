"""
Four-Layer ERC Engine - Chip Compatibility Verification via Hardware Semantic Reasoning.

This module implements the core reasoning engine for VeriQuery's ERC
(Electrical Rule Check) system. It employs a four-layer progressive
detection architecture:

  Layer 1 - Static Stability: Logic level compatibility (JEDEC-based)
  Layer 2 - Signal Integrity: Transmission line reflection analysis
  Layer 3 - Topology Conflict: Interface protocol and port attribute checks
  Layer 4 - Environmental Degradation: Temperature drift and process aging

Academic References:
  1. JEDEC JESD8 series - Logic level definitions (Layer 1)
  2. IEEE 1801 UPF - Multi-voltage domain design (Layer 1, 3)
  3. Bogatin, E. "Signal and Power Integrity - Simplified" (Layer 2)
  4. Moore, R. E. (1966). "Interval Analysis" (Layer 4)
  5. JEDEC JESD22-A108D - Temperature, Bias and Operating Life (Layer 4)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from core import (
    get_settings, ElectricalSpec, Citation,
    ERCSeverity
)


@dataclass
class Interval:
    """Interval arithmetic primitive for uncertainty propagation.

    Models an uncertain quantity as [lo, hi] and propagates uncertainty
    through arithmetic operations. Used in Layer 4 for temperature drift
    modeling where electrical parameters are expanded from crisp values
    to intervals accounting for thermal variation.

    Key operations:
        [a,b] - [c,d] = [a-d, b-c]   (worst-case subtraction)
        [a,b] * k     = [a*k, b*k]     (k >= 0)
        [a,b] * k     = [b*k, a*k]     (k < 0, endpoints swap)

    Attributes:
        lo: Lower bound of the interval.
        hi: Upper bound of the interval.
    """
    lo: float
    hi: float

    def __post_init__(self):
        if self.lo > self.hi:
            self.lo, self.hi = self.hi, self.lo

    def __sub__(self, other: 'Interval') -> 'Interval':
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __mul__(self, k: float) -> 'Interval':
        if k >= 0:
            return Interval(self.lo * k, self.hi * k)
        return Interval(self.hi * k, self.lo * k)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def mid(self) -> float:
        return (self.lo + self.hi) / 2

    @classmethod
    def from_crisp_with_drift(cls, value: float, drift: float) -> 'Interval':
        return cls(value - abs(drift), value + abs(drift))

    def definitely_positive(self) -> bool:
        return self.lo > 0

    def possibly_negative(self) -> bool:
        return self.hi < 0


logger = logging.getLogger(__name__)

_cached_layer_rules = None
_shared_degradation_db = None


@dataclass
class ERCRuleResult:
    """Result of a single ERC rule check.

    Attributes:
        rule_id: Unique rule identifier (e.g. "ERC-L1-V001").
        rule_name: Human-readable rule name.
        passed: Whether the rule check passed.
        severity: Severity level (ERROR / WARNING / INFO).
        message: Human-readable check result description.
        actual_values: Measured parameter values.
        expected_condition: Expected condition description.
        margin: Safety margin (positive = safe, negative = violation).
        suggestion: Improvement suggestion when check fails.
    """
    rule_id: str
    rule_name: str
    passed: bool
    severity: ERCSeverity
    message: str
    actual_values: Dict[str, float] = field(default_factory=dict)
    expected_condition: str = ""
    margin: Optional[float] = None
    suggestion: str = ""


class ERCLayer(Enum):
    """Enumeration of the four ERC detection layers."""
    LAYER1_STATIC = "layer1_static"
    LAYER2_REFLECTION = "layer2_reflection"
    LAYER3_TOPOLOGY = "layer3_topology"
    LAYER4_ENVIRONMENT = "layer4_environment"


@dataclass
class LayerCheckResult:
    """Aggregated result for a single ERC layer.

    Attributes:
        layer: Layer identifier enum.
        layer_name: Human-readable layer name.
        passed: Whether the layer passed overall.
        severity: Most severe level in this layer.
        results: List of individual rule results.
        summary: Layer-level summary description.
        metadata: Layer-level metadata (chip names, error counts, etc.).
    """
    layer: ERCLayer
    layer_name: str
    passed: bool
    severity: ERCSeverity
    results: List[ERCRuleResult]
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FourLayerERCResult:
    """Final result of the four-layer ERC check.

    Attributes:
        layer1_result: Layer 1 (static stability) result.
        layer2_result: Layer 2 (signal integrity) result.
        layer3_result: Layer 3 (topology conflict) result.
        layer4_result: Layer 4 (environmental degradation) result.
        overall_compatible: Overall compatibility verdict.
        overall_confidence: Confidence score (0.0 - 1.0).
        summary: Overall summary description.
        suggestions: List of improvement suggestions.
        citations: Source citations for traceability.
    """
    layer1_result: Optional[LayerCheckResult] = None
    layer2_result: Optional[LayerCheckResult] = None
    layer3_result: Optional[LayerCheckResult] = None
    layer4_result: Optional[LayerCheckResult] = None
    overall_compatible: bool = False
    overall_confidence: float = 0.0
    summary: str = ""
    suggestions: List[str] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)


class ProcessType(Enum):
    """Semiconductor process type enumeration."""
    BICMOS = "BiCMOS"
    CMOS = "CMOS"
    NMOS = "NMOS"
    PMOS = "PMOS"
    TTL = "TTL"
    ECL = "ECL"
    UNKNOWN = "UNKNOWN"


class PackageType(Enum):
    """IC package type enumeration."""
    DIP = "DIP"
    SOP = "SOP"
    QFP = "QFP"
    BGA = "BGA"
    LGA = "LGA"
    CSP = "CSP"
    UNKNOWN = "UNKNOWN"


@dataclass
class DegradationData:
    """Degradation parameters for a specific process + package combination.

    Attributes:
        process_type: Process type (CMOS / BiCMOS / TTL etc.).
        package_type: Package type (DIP / SOP / BGA etc.).
        temperature_coefficient: Degradation increment per °C.
        humidity_coefficient: Degradation increment per %RH.
        voltage_coefficient: Degradation increment per V.
        base_degradation_rate: Degradation rate per 1000 hours at reference conditions.
        activation_energy: Arrhenius activation energy in eV.
        confidence_interval: Statistical confidence bounds for the degradation factor.
        reference_temperature: Reference temperature in °C.
        reference_humidity: Reference humidity in %RH.
        reference_voltage: Reference voltage in V.
    """
    process_type: ProcessType
    package_type: PackageType
    temperature_coefficient: float
    humidity_coefficient: float
    voltage_coefficient: float
    base_degradation_rate: float
    activation_energy: float
    confidence_interval: Tuple[float, float]
    reference_temperature: float = 25.0
    reference_humidity: float = 60.0
    reference_voltage: float = 3.3


@dataclass
class DegradationResult:
    """Result of process degradation assessment.

    Attributes:
        degradation_factor: Retained parameter ratio (1.0 = no degradation, 0.5 = 50% degraded).
        confidence_interval: Statistical confidence bounds for the degradation factor.
        degradation_rate: Combined degradation rate with Arrhenius acceleration.
        estimated_lifetime: Estimated time to 50% degradation in hours.
        risk_level: Risk classification (LOW / MEDIUM / HIGH / CRITICAL).
        recommendations: Improvement suggestions.
    """
    degradation_factor: float
    confidence_interval: Tuple[float, float]
    degradation_rate: float
    estimated_lifetime: float
    risk_level: str
    recommendations: List[str]


class ProcessDegradationDatabase:
    """Process degradation reference database based on JEDEC JESD22 and Arrhenius equation.

    Stores degradation parameters for 18 process + package combinations and
    computes degradation factors using the Arrhenius acceleration model.
    """

    def __init__(self):
        self.degradation_data = self._initialize_degradation_data()
        self.boltzmann_constant = 8.617e-5
        logger.info("Process degradation database initialized")

    def _initialize_degradation_data(self) -> Dict[str, DegradationData]:
        return {
            "CMOS_DIP": DegradationData(
                process_type=ProcessType.CMOS, package_type=PackageType.DIP,
                temperature_coefficient=0.002, humidity_coefficient=0.001,
                voltage_coefficient=0.001, base_degradation_rate=0.1,
                activation_energy=0.7, confidence_interval=(0.95, 1.05)
            ),
            "CMOS_SOP": DegradationData(
                process_type=ProcessType.CMOS, package_type=PackageType.SOP,
                temperature_coefficient=0.0025, humidity_coefficient=0.0015,
                voltage_coefficient=0.0012, base_degradation_rate=0.12,
                activation_energy=0.7, confidence_interval=(0.93, 1.07)
            ),
            "CMOS_QFP": DegradationData(
                process_type=ProcessType.CMOS, package_type=PackageType.QFP,
                temperature_coefficient=0.003, humidity_coefficient=0.002,
                voltage_coefficient=0.0015, base_degradation_rate=0.15,
                activation_energy=0.7, confidence_interval=(0.90, 1.10)
            ),
            "CMOS_BGA": DegradationData(
                process_type=ProcessType.CMOS, package_type=PackageType.BGA,
                temperature_coefficient=0.002, humidity_coefficient=0.001,
                voltage_coefficient=0.001, base_degradation_rate=0.08,
                activation_energy=0.7, confidence_interval=(0.96, 1.04)
            ),
            "CMOS_LGA": DegradationData(
                process_type=ProcessType.CMOS, package_type=PackageType.LGA,
                temperature_coefficient=0.0022, humidity_coefficient=0.0012,
                voltage_coefficient=0.0011, base_degradation_rate=0.09,
                activation_energy=0.7, confidence_interval=(0.95, 1.05)
            ),
            "CMOS_CSP": DegradationData(
                process_type=ProcessType.CMOS, package_type=PackageType.CSP,
                temperature_coefficient=0.0018, humidity_coefficient=0.0009,
                voltage_coefficient=0.0009, base_degradation_rate=0.07,
                activation_energy=0.7, confidence_interval=(0.97, 1.03)
            ),
            "BiCMOS_DIP": DegradationData(
                process_type=ProcessType.BICMOS, package_type=PackageType.DIP,
                temperature_coefficient=0.0015, humidity_coefficient=0.0008,
                voltage_coefficient=0.0008, base_degradation_rate=0.08,
                activation_energy=0.8, confidence_interval=(0.97, 1.03)
            ),
            "BiCMOS_SOP": DegradationData(
                process_type=ProcessType.BICMOS, package_type=PackageType.SOP,
                temperature_coefficient=0.0018, humidity_coefficient=0.001,
                voltage_coefficient=0.001, base_degradation_rate=0.09,
                activation_energy=0.8, confidence_interval=(0.95, 1.05)
            ),
            "BiCMOS_QFP": DegradationData(
                process_type=ProcessType.BICMOS, package_type=PackageType.QFP,
                temperature_coefficient=0.002, humidity_coefficient=0.0012,
                voltage_coefficient=0.001, base_degradation_rate=0.1,
                activation_energy=0.8, confidence_interval=(0.94, 1.06)
            ),
            "BiCMOS_BGA": DegradationData(
                process_type=ProcessType.BICMOS, package_type=PackageType.BGA,
                temperature_coefficient=0.0015, humidity_coefficient=0.0008,
                voltage_coefficient=0.0008, base_degradation_rate=0.07,
                activation_energy=0.8, confidence_interval=(0.97, 1.03)
            ),
            "TTL_DIP": DegradationData(
                process_type=ProcessType.TTL, package_type=PackageType.DIP,
                temperature_coefficient=0.001, humidity_coefficient=0.0005,
                voltage_coefficient=0.0005, base_degradation_rate=0.05,
                activation_energy=0.9, confidence_interval=(0.98, 1.02)
            ),
            "TTL_SOP": DegradationData(
                process_type=ProcessType.TTL, package_type=PackageType.SOP,
                temperature_coefficient=0.0012, humidity_coefficient=0.0006,
                voltage_coefficient=0.0006, base_degradation_rate=0.06,
                activation_energy=0.9, confidence_interval=(0.97, 1.03)
            ),
            "NMOS_DIP": DegradationData(
                process_type=ProcessType.NMOS, package_type=PackageType.DIP,
                temperature_coefficient=0.0025, humidity_coefficient=0.0015,
                voltage_coefficient=0.0015, base_degradation_rate=0.12,
                activation_energy=0.65, confidence_interval=(0.92, 1.08)
            ),
            "NMOS_QFP": DegradationData(
                process_type=ProcessType.NMOS, package_type=PackageType.QFP,
                temperature_coefficient=0.003, humidity_coefficient=0.002,
                voltage_coefficient=0.0018, base_degradation_rate=0.15,
                activation_energy=0.65, confidence_interval=(0.90, 1.10)
            ),
            "PMOS_DIP": DegradationData(
                process_type=ProcessType.PMOS, package_type=PackageType.DIP,
                temperature_coefficient=0.0028, humidity_coefficient=0.0018,
                voltage_coefficient=0.0016, base_degradation_rate=0.14,
                activation_energy=0.62, confidence_interval=(0.91, 1.09)
            ),
            "ECL_DIP": DegradationData(
                process_type=ProcessType.ECL, package_type=PackageType.DIP,
                temperature_coefficient=0.0008, humidity_coefficient=0.0004,
                voltage_coefficient=0.0004, base_degradation_rate=0.04,
                activation_energy=1.0, confidence_interval=(0.98, 1.02)
            ),
            "ECL_QFP": DegradationData(
                process_type=ProcessType.ECL, package_type=PackageType.QFP,
                temperature_coefficient=0.001, humidity_coefficient=0.0005,
                voltage_coefficient=0.0005, base_degradation_rate=0.05,
                activation_energy=1.0, confidence_interval=(0.97, 1.03)
            ),
            "ECL_BGA": DegradationData(
                process_type=ProcessType.ECL, package_type=PackageType.BGA,
                temperature_coefficient=0.0007, humidity_coefficient=0.0003,
                voltage_coefficient=0.0003, base_degradation_rate=0.03,
                activation_energy=1.0, confidence_interval=(0.99, 1.01)
            ),
        }

    def get_degradation_data(
        self,
        process_type: ProcessType,
        package_type: PackageType
    ) -> Optional[DegradationData]:
        key = f"{process_type.value}_{package_type.value}"
        return self.degradation_data.get(key)

    def calculate_degradation_factor(
        self,
        process_type: ProcessType,
        package_type: PackageType,
        temperature: float = 25.0,
        humidity: float = 60.0,
        voltage: float = 3.3,
        operating_hours: float = 1000.0
    ) -> DegradationResult:
        """Calculate process degradation factor using Arrhenius acceleration model.

        Algorithm:
            1. AF = exp(Ea/kB * (1/T_ref - 1/T))   (Arrhenius acceleration)
            2. Linear drifts for temperature, humidity, voltage deviations
            3. total_rate = base_rate * AF * (1 + drifts)
            4. factor = 1.0 - total_rate * hours / 1000, clamped to [0.1, 1.0]

        Args:
            process_type: Semiconductor process type.
            package_type: IC package type.
            temperature: Operating temperature in °C.
            humidity: Relative humidity in %RH.
            voltage: Operating voltage in V.
            operating_hours: Operating time in hours.

        Returns:
            DegradationResult with factor, confidence interval, lifetime, and risk.
        """
        degradation_data = self.get_degradation_data(process_type, package_type)

        if not degradation_data:
            degradation_data = self.degradation_data.get("CMOS_DIP")

        if not degradation_data:
            return DegradationResult(
                degradation_factor=1.0,
                confidence_interval=(0.9, 1.1),
                degradation_rate=0.1,
                estimated_lifetime=100000,
                risk_level="UNKNOWN",
                recommendations=["Missing process degradation data"]
            )

        temp_kelvin = temperature + 273.15
        ref_temp_kelvin = degradation_data.reference_temperature + 273.15

        arrhenius_factor = math.exp(
            (degradation_data.activation_energy / self.boltzmann_constant) *
            (1 / ref_temp_kelvin - 1 / temp_kelvin)
        )

        temp_drift = degradation_data.temperature_coefficient * (temperature - degradation_data.reference_temperature)
        humidity_drift = degradation_data.humidity_coefficient * (humidity - degradation_data.reference_humidity)
        voltage_drift = degradation_data.voltage_coefficient * (voltage - degradation_data.reference_voltage)

        total_degradation_rate = (
            degradation_data.base_degradation_rate *
            arrhenius_factor *
            (1 + temp_drift + humidity_drift + voltage_drift)
        )

        degradation_factor = 1.0 - (total_degradation_rate * operating_hours / 1000.0)
        degradation_factor = max(degradation_factor, 0.1)

        confidence_lo = degradation_factor * degradation_data.confidence_interval[0]
        confidence_hi = degradation_factor * degradation_data.confidence_interval[1]

        if total_degradation_rate > 0:
            estimated_lifetime = (1.0 - 0.5) / (total_degradation_rate / 1000.0)
        else:
            estimated_lifetime = float('inf')

        risk_level = self._assess_risk_level(degradation_factor, temperature)
        recommendations = self._generate_recommendations(
            degradation_factor, temperature, humidity, voltage
        )

        return DegradationResult(
            degradation_factor=degradation_factor,
            confidence_interval=(confidence_lo, confidence_hi),
            degradation_rate=total_degradation_rate,
            estimated_lifetime=estimated_lifetime,
            risk_level=risk_level,
            recommendations=recommendations
        )

    def _assess_risk_level(self, degradation_factor: float, temperature: float) -> str:
        """Assess risk level based on degradation factor and temperature.

        Uses a 2D decision matrix: degradation factor vs. temperature.
        Temperature thresholds: 85°C (industrial), 105°C (extended).
        """
        if degradation_factor >= 0.95:
            if temperature <= 85:
                return "LOW"
            elif temperature <= 105:
                return "MEDIUM"
            else:
                return "HIGH"
        elif degradation_factor >= 0.85:
            if temperature <= 85:
                return "MEDIUM"
            else:
                return "HIGH"
        elif degradation_factor >= 0.5:
            return "HIGH"
        else:
            return "CRITICAL"

    def _generate_recommendations(
        self,
        degradation_factor: float,
        temperature: float,
        humidity: float,
        voltage: float
    ) -> List[str]:
        """Generate engineering recommendations based on degradation assessment."""
        recommendations = []

        if degradation_factor < 0.5:
            recommendations.append("退化因子极低，器件可能已超出安全工作区，强烈建议更换或降额使用")
        elif degradation_factor < 0.85:
            recommendations.append("退化因子较低，建议降低工作温度或增加散热设计")

        if temperature > 105:
            recommendations.append("工作温度过高，建议使用高温等级器件")
        elif temperature > 85:
            recommendations.append("工作温度较高，建议增加散热措施")

        if humidity > 85:
            recommendations.append("工作湿度较高，建议增加防潮措施")

        if voltage > 5.0:
            recommendations.append("工作电压较高，建议检查器件电压耐受能力")

        if not recommendations:
            recommendations.append("工作条件良好，建议定期维护")

        return recommendations

    def parse_process_type(self, process_str: str) -> ProcessType:
        """Parse a string into a ProcessType enum.

        Strategy: exact match first (O(1)), then fuzzy substring match (O(n)),
        finally return UNKNOWN.
        """
        process_str_upper = process_str.upper()
        exact_map = {
            "BICMOS": ProcessType.BICMOS, "CMOS": ProcessType.CMOS,
            "NMOS": ProcessType.NMOS, "PMOS": ProcessType.PMOS,
            "TTL": ProcessType.TTL, "ECL": ProcessType.ECL,
        }
        if process_str_upper in exact_map:
            return exact_map[process_str_upper]
        for process_type in ProcessType:
            if process_type.value.upper() in process_str_upper:
                return process_type
        return ProcessType.UNKNOWN

    def parse_package_type(self, package_str: str) -> PackageType:
        """Parse a string into a PackageType enum.

        Strategy: exact match first (O(1)), then fuzzy substring match (O(n)),
        finally return UNKNOWN.
        """
        package_str_upper = package_str.upper()
        exact_map = {
            "DIP": PackageType.DIP, "SOP": PackageType.SOP,
            "QFP": PackageType.QFP, "BGA": PackageType.BGA,
            "LGA": PackageType.LGA, "CSP": PackageType.CSP,
        }
        if package_str_upper in exact_map:
            return exact_map[package_str_upper]
        for package_type in PackageType:
            if package_type.value.upper() in package_str_upper:
                return package_type
        return PackageType.UNKNOWN


class FourLayerERCEngine:
    """Four-layer ERC engine for chip compatibility verification.

    Architecture:
        Layer 1 (Static Stability) -> Layer 2 (Signal Integrity)
        -> Layer 3 (Topology Conflict) -> Layer 4 (Environmental Degradation)

    Layer 1 and Layer 3 are hard requirements (ERROR on failure).
    Layer 2 and Layer 4 are advisory (WARNING on failure).

    Usage:
        engine = FourLayerERCEngine()
        result = engine.check_four_layer(
            driver_chip="SN74HC04",
            receiver_chip="SN74HCT04",
            driver_params={"VCC": 5.0, "VOH": 4.9, ...},
            receiver_params={"VCC": 5.0, "VIH": 3.5, ...},
        )
    """

    DEFAULT_TEMPERATURE = 25.0
    DEFAULT_HUMIDITY = 60.0
    DEFAULT_OPERATING_HOURS = 10000
    MIN_MARGIN_THRESHOLD = 0.1

    def __init__(self, settings=None):
        """Initialize the four-layer ERC engine.

        Args:
            settings: Global configuration object. Defaults to get_settings().
        """
        global _cached_layer_rules, _shared_degradation_db
        self.settings = settings or get_settings()

        if _cached_layer_rules is None:
            _cached_layer_rules = {
                1: self._define_layer1_rules(),
                2: self._define_layer2_rules(),
                3: self._define_layer3_rules(),
                4: self._define_layer4_rules(),
            }
        self.layer1_rules = _cached_layer_rules[1]
        self.layer2_rules = _cached_layer_rules[2]
        self.layer3_rules = _cached_layer_rules[3]
        self.layer4_rules = _cached_layer_rules[4]

        if _shared_degradation_db is None:
            _shared_degradation_db = ProcessDegradationDatabase()
        self.degradation_db = _shared_degradation_db
        logger.info("Four-layer ERC engine initialized")

    def check_four_layer(
        self,
        driver_chip: str,
        receiver_chip: str,
        driver_params: Dict[str, float],
        receiver_params: Dict[str, float],
        citations: List[Citation] = None,
        trace_length: Optional[float] = None,
        trace_impedance: Optional[float] = None,
        temperature: Optional[float] = None,
        topology_info: Optional[Dict[str, Any]] = None
    ) -> FourLayerERCResult:
        """Execute the four-layer ERC check.

        Runs all four layers sequentially (no short-circuit) so the user
        always receives a complete report.

        Args:
            driver_chip: Driver chip name (for display only).
            receiver_chip: Receiver chip name (for display only).
            driver_params: Driver electrical parameters dict. Supported keys:
                VCC/supply_voltage, VOH, VOL, IOH, IOL, rise_time/tr,
                fall_time/tf, output_impedance/Zo, process/family, package.
            receiver_params: Receiver electrical parameters dict. Supported keys:
                VCC/supply_voltage, VIH, VIL, IIH, IIL, input_impedance/Zi,
                process/family, package.
            citations: Source citations for traceability.
            trace_length: Trace length in cm (for Layer 2).
            trace_impedance: Trace impedance in Ω (for Layer 2).
            temperature: Operating temperature in °C (for Layer 4).
            topology_info: Topology info dict (for Layer 3). Supported keys:
                driver_interface, receiver_interface, driver_direction,
                receiver_direction, driver_pull, receiver_pull,
                driver_tri_state, driver_voltage, receiver_voltage.

        Returns:
            FourLayerERCResult with per-layer results, overall compatibility,
            confidence, summary, and suggestions.
        """
        logger.info(f"Four-layer ERC check: {driver_chip} -> {receiver_chip}")

        citations = citations or []
        result = FourLayerERCResult()

        result.layer1_result = self._check_layer1_static_stability(
            driver_chip, receiver_chip, driver_params, receiver_params
        )
        result.layer2_result = self._check_layer2_reflection_warning(
            driver_chip, receiver_chip, driver_params, receiver_params,
            trace_length, trace_impedance
        )
        result.layer3_result = self._check_layer3_topology_conflict(
            driver_chip, receiver_chip, driver_params, receiver_params,
            topology_info
        )
        result.layer4_result = self._check_layer4_extreme_environment(
            driver_chip, receiver_chip, driver_params, receiver_params,
            temperature
        )

        result.overall_compatible = self._evaluate_overall_compatibility(result)
        result.overall_confidence = self._calculate_overall_confidence(result)
        result.summary = self._generate_overall_summary(result)
        result.suggestions = self._generate_suggestions(result)
        result.citations = citations

        logger.info(
            f"Four-layer ERC check complete: compatible={result.overall_compatible}, "
            f"confidence={result.overall_confidence:.2f}"
        )
        return result

    def _check_layer1_static_stability(
        self,
        driver_chip: str,
        receiver_chip: str,
        driver_params: Dict[str, float],
        receiver_params: Dict[str, float]
    ) -> LayerCheckResult:
        """Layer 1: Static stability detection.

        Checks logic level compatibility based on JEDEC standards:
        supply voltage, high/low level, RNI, drive capability, fanout.

        Pass condition: no ERROR-level rule failures.
        """
        logger.info("Layer 1: 静态稳定性检测")

        results = []
        for rule in self.layer1_rules:
            check_result = rule["check_fn"](driver_params, receiver_params)
            if check_result is not None:
                results.append(check_result)

        errors = [r for r in results if r.severity == ERCSeverity.ERROR and not r.passed]
        warnings = [r for r in results if r.severity == ERCSeverity.WARNING and not r.passed]
        passed = len(errors) == 0
        severity = ERCSeverity.INFO if passed else ERCSeverity.ERROR

        return LayerCheckResult(
            layer=ERCLayer.LAYER1_STATIC,
            layer_name="静态稳定性检测",
            passed=passed,
            severity=severity,
            results=results,
            summary=self._generate_layer_summary(1, passed, results, driver_chip, receiver_chip),
            metadata={
                "driver_chip": driver_chip,
                "receiver_chip": receiver_chip,
                "errors_count": len(errors),
                "warnings_count": len(warnings),
            }
        )

    def _check_layer2_reflection_warning(
        self,
        driver_chip: str,
        receiver_chip: str,
        driver_params: Dict[str, float],
        receiver_params: Dict[str, float],
        trace_length: Optional[float],
        trace_impedance: Optional[float]
    ) -> LayerCheckResult:
        """Layer 2: Quasi-physical reflection warning.

        Checks transmission line length and reflection coefficient.

        Pass condition: no WARNING-level rule failures.
        """
        logger.info("Layer 2: 准物理反射预警")

        results = []
        for rule in self.layer2_rules:
            check_result = rule["check_fn"](driver_params, receiver_params, trace_length, trace_impedance)
            if check_result is not None:
                results.append(check_result)

        warnings = [r for r in results if r.severity == ERCSeverity.WARNING and not r.passed]
        passed = len(warnings) == 0
        severity = ERCSeverity.INFO if passed else ERCSeverity.WARNING

        return LayerCheckResult(
            layer=ERCLayer.LAYER2_REFLECTION,
            layer_name="准物理反射预警",
            passed=passed,
            severity=severity,
            results=results,
            summary=self._generate_layer_summary(2, passed, results, driver_chip, receiver_chip),
            metadata={
                "driver_chip": driver_chip,
                "receiver_chip": receiver_chip,
                "trace_length": trace_length,
                "trace_impedance": trace_impedance,
                "warnings_count": len(warnings),
            }
        )

    def _check_layer3_topology_conflict(
        self,
        driver_chip: str,
        receiver_chip: str,
        driver_params: Dict[str, float],
        receiver_params: Dict[str, float],
        topology_info: Optional[Dict[str, Any]]
    ) -> LayerCheckResult:
        """Layer 3: Topology conflict arbitration.

        Checks interface protocol compatibility and port attribute conflicts.

        Pass condition: no ERROR-level rule failures.
        """
        logger.info("Layer 3: 拓扑冲突仲裁")

        results = []
        for rule in self.layer3_rules:
            result = rule["check_fn"](driver_params, receiver_params, topology_info)
            if result:
                results.append(result)

        errors = [r for r in results if r.severity == ERCSeverity.ERROR and not r.passed]
        warnings = [r for r in results if r.severity == ERCSeverity.WARNING and not r.passed]
        passed = len(errors) == 0
        severity = ERCSeverity.INFO if passed else ERCSeverity.ERROR

        return LayerCheckResult(
            layer=ERCLayer.LAYER3_TOPOLOGY,
            layer_name="拓扑冲突仲裁",
            passed=passed,
            severity=severity,
            results=results,
            summary=self._generate_layer_summary(3, passed, results, driver_chip, receiver_chip),
            metadata={
                "driver_chip": driver_chip,
                "receiver_chip": receiver_chip,
                "topology_info": topology_info,
                "errors_count": len(errors),
                "warnings_count": len(warnings),
            }
        )

    def _check_layer4_extreme_environment(
        self,
        driver_chip: str,
        receiver_chip: str,
        driver_params: Dict[str, float],
        receiver_params: Dict[str, float],
        temperature: Optional[float]
    ) -> LayerCheckResult:
        """Layer 4: Extreme environment degradation assessment.

        Checks temperature drift (interval arithmetic) and process degradation
        (Arrhenius model).

        Pass condition: no WARNING-level rule failures.
        """
        logger.info("Layer 4: 极端环境退化评估")

        results = []
        for rule in self.layer4_rules:
            check_result = rule["check_fn"](
                driver_params, receiver_params,
                temperature or self.DEFAULT_TEMPERATURE
            )
            if check_result is not None:
                results.append(check_result)

        warnings = [r for r in results if r.severity == ERCSeverity.WARNING and not r.passed]
        passed = len(warnings) == 0
        severity = ERCSeverity.INFO if passed else ERCSeverity.WARNING

        return LayerCheckResult(
            layer=ERCLayer.LAYER4_ENVIRONMENT,
            layer_name="极端环境退化评估",
            passed=passed,
            severity=severity,
            results=results,
            summary=self._generate_layer_summary(4, passed, results, driver_chip, receiver_chip),
            metadata={
                "driver_chip": driver_chip,
                "receiver_chip": receiver_chip,
                "temperature": temperature,
                "warnings_count": len(warnings),
            }
        )

    def _define_layer1_rules(self) -> List[Dict]:
        return [
            {
                "rule_id": "ERC-L1-V000",
                "rule_name": "供电电压兼容性",
                "description": "|VCC_driver - VCC_receiver| <= 0.3V (IEEE 1801 UPF)",
                "severity": ERCSeverity.ERROR,
                "check_fn": self._check_supply_voltage
            },
            {
                "rule_id": "ERC-L1-V001",
                "rule_name": "高电平兼容性",
                "description": "VOH >= VIH (JEDEC JESD8)",
                "severity": ERCSeverity.ERROR,
                "check_fn": self._check_high_level
            },
            {
                "rule_id": "ERC-L1-V002",
                "rule_name": "低电平兼容性",
                "description": "VOL <= VIL (JEDEC JESD8)",
                "severity": ERCSeverity.ERROR,
                "check_fn": self._check_low_level
            },
            {
                "rule_id": "ERC-L1-N001",
                "rule_name": "归一化噪声免疫系数（RNI）",
                "description": "RNI >= 0.2",
                "severity": ERCSeverity.WARNING,
                "check_fn": self._check_rni_coefficient
            },
            {
                "rule_id": "ERC-L1-I001",
                "rule_name": "高电平驱动能力",
                "description": "|IOH| >= |IIH|",
                "severity": ERCSeverity.ERROR,
                "check_fn": self._check_high_drive_capability
            },
            {
                "rule_id": "ERC-L1-I002",
                "rule_name": "低电平驱动能力",
                "description": "IOL >= |IIL|",
                "severity": ERCSeverity.ERROR,
                "check_fn": self._check_low_drive_capability
            },
            {
                "rule_id": "ERC-L1-I003",
                "rule_name": "扇出能力评估",
                "description": "Fanout >= 1",
                "severity": ERCSeverity.INFO,
                "check_fn": self._check_fanout
            },
        ]

    def _define_layer2_rules(self) -> List[Dict]:
        return [
            {
                "rule_id": "ERC-L2-T001",
                "rule_name": "临界传输线长度计算",
                "description": "trace_length < critical_length",
                "severity": ERCSeverity.WARNING,
                "check_fn": self._check_critical_length
            },
            {
                "rule_id": "ERC-L2-R001",
                "rule_name": "反射系数分析",
                "description": "|Γ| < 0.15",
                "severity": ERCSeverity.WARNING,
                "check_fn": self._check_reflection_coefficient
            },
        ]

    def _define_layer3_rules(self) -> List[Dict]:
        return [
            {
                "rule_id": "ERC-L3-I001",
                "rule_name": "接口协议兼容性检查",
                "description": "接口契约兼容 (IEEE 1801 UPF)",
                "severity": ERCSeverity.ERROR,
                "check_fn": self._check_interface_contract
            },
            {
                "rule_id": "ERC-L3-P001",
                "rule_name": "端口属性冲突检测",
                "description": "端口属性无冲突 (Formal Verification)",
                "severity": ERCSeverity.ERROR,
                "check_fn": self._check_pam_conflict
            },
        ]

    def _define_layer4_rules(self) -> List[Dict]:
        return [
            {
                "rule_id": "ERC-L4-T001",
                "rule_name": "温度漂移检测",
                "description": "Temperature drift within acceptable range",
                "severity": ERCSeverity.WARNING,
                "check_fn": self._check_temperature_drift
            },
            {
                "rule_id": "ERC-L4-P001",
                "rule_name": "工艺退化基准库查询",
                "description": "Process degradation within acceptable range",
                "severity": ERCSeverity.WARNING,
                "check_fn": self._check_process_degradation
            },
        ]

    def _check_supply_voltage(self, driver: Dict, receiver: Dict) -> Optional[ERCRuleResult]:
        """Check supply voltage compatibility [ERC-L1-V000].

        Rule: |VCC_driver - VCC_receiver| <= 0.3V
        Reference: IEEE 1801 UPF multi-voltage domain design
        """
        driver_vcc = driver.get('VCC') or driver.get('supply_voltage')
        receiver_vcc = receiver.get('VCC') or receiver.get('supply_voltage')

        if driver_vcc is None or receiver_vcc is None:
            return None

        voltage_diff = abs(driver_vcc - receiver_vcc)
        passed = voltage_diff <= 0.3

        return ERCRuleResult(
            rule_id='ERC-L1-V000',
            rule_name='供电电压兼容性',
            passed=passed,
            severity=ERCSeverity.ERROR if not passed else ERCSeverity.INFO,
            message=f'电压差={voltage_diff:.2f}V ({"通过" if passed else "失败"})',
            actual_values={'driver_vcc': driver_vcc, 'receiver_vcc': receiver_vcc},
            expected_condition='|VCC_driver - VCC_receiver| <= 0.3V',
            margin=0.3 - voltage_diff,
        )

    def _check_high_level(self, driver: Dict, receiver: Dict) -> Optional[ERCRuleResult]:
        """Check high-level compatibility [ERC-L1-V001]: VOH >= VIH."""
        if "VOH" not in driver or "VIH" not in receiver:
            return None

        voh = driver["VOH"]
        vih = receiver["VIH"]
        passed = voh >= vih
        margin = voh - vih
        severity = ERCSeverity.INFO if passed else ERCSeverity.ERROR

        return ERCRuleResult(
            rule_id="ERC-L1-V001",
            rule_name="高电平兼容性",
            passed=passed,
            severity=severity,
            message=f"VOH({voh}V) {'>' if margin > 0 else '='} VIH({vih}V), 余量={margin:.3f}V",
            actual_values={"VOH": voh, "VIH": vih},
            expected_condition="VOH >= VIH",
            margin=margin
        )

    def _check_low_level(self, driver: Dict, receiver: Dict) -> Optional[ERCRuleResult]:
        """Check low-level compatibility [ERC-L1-V002]: VOL <= VIL."""
        if "VOL" not in driver or "VIL" not in receiver:
            return None

        vol = driver["VOL"]
        vil = receiver["VIL"]
        passed = vol <= vil
        margin = vil - vol
        severity = ERCSeverity.INFO if passed else ERCSeverity.ERROR

        return ERCRuleResult(
            rule_id="ERC-L1-V002",
            rule_name="低电平兼容性",
            passed=passed,
            severity=severity,
            message=f"VOL({vol}V) {'<' if margin > 0 else '='} VIL({vil}V), 余量={margin:.3f}V",
            actual_values={"VOL": vol, "VIL": vil},
            expected_condition="VOL <= VIL",
            margin=margin
        )

    def _check_rni_coefficient(self, driver: Dict, receiver: Dict) -> Optional[ERCRuleResult]:
        """Check Relative Noise Immunity [ERC-L1-N001]: RNI >= 0.2.

        RNI = min(VOH - VIH, VIL - VOL) / (VOH - VOL)
        """
        voh = driver.get("VOH")
        vol = driver.get("VOL")
        vih = receiver.get("VIH")
        vil = receiver.get("VIL")

        if None in [voh, vol, vih, vil]:
            return None

        high_margin = voh - vih
        low_margin = vil - vol
        logic_swing = voh - vol

        if logic_swing == 0:
            return None

        rni = min(high_margin, low_margin) / logic_swing
        passed = rni >= 0.2
        quality = "优秀" if rni >= 0.3 else ("良好" if rni >= 0.2 else "不足")

        return ERCRuleResult(
            rule_id="ERC-L1-N001",
            rule_name="归一化噪声免疫系数（RNI）",
            passed=passed,
            severity=ERCSeverity.WARNING if not passed else ERCSeverity.INFO,
            message=f"RNI={rni:.3f} ({quality})",
            actual_values={"RNI": rni, "high_margin": high_margin, "low_margin": low_margin},
            expected_condition="RNI >= 0.2 (推荐>=0.3)",
            margin=rni - 0.2
        )

    def _normalize_current_to_amps(self, value: float, unit: Optional[str]) -> float:
        if not unit:
            return value
        u = unit.strip().replace("µ", "u").replace("μ", "u").upper()
        if u in ["MA", "MAMP"]:
            return value / 1000
        elif u in ["UA", "UAMP"]:
            return value / 1000000
        elif u in ["NA", "NAMP"]:
            return value / 1000000000
        elif u in ["A", "AMP"]:
            return value
        return value

    def _format_current_value(self, value: float, unit: str) -> str:
        """Format current value with auto-ranging (A / mA / µA)."""
        value_amps = self._normalize_current_to_amps(value, unit)

        if abs(value_amps) >= 1:
            return f"{abs(value_amps):.3f}A"
        elif abs(value_amps) >= 0.001:
            return f"{abs(value_amps * 1000):.3f}mA"
        else:
            return f"{abs(value_amps * 1000000):.3f}µA"

    def _check_high_drive_capability(self, driver: Dict, receiver: Dict) -> Optional[ERCRuleResult]:
        """Check high-level drive capability [ERC-L1-I001]: |IOH| >= |IIH|."""
        if "IOH" not in driver or "IIH" not in receiver:
            return None

        ioh = driver["IOH"]
        iih = receiver["IIH"]
        ioh_unit = driver.get("IOH_unit", "mA")
        iih_unit = receiver.get("IIH_unit", "mA")

        ioh_amps = self._normalize_current_to_amps(ioh, ioh_unit)
        iih_amps = self._normalize_current_to_amps(iih, iih_unit)

        passed = abs(ioh_amps) >= abs(iih_amps)
        margin = abs(ioh_amps) - abs(iih_amps)

        return ERCRuleResult(
            rule_id="ERC-L1-I001",
            rule_name="高电平驱动能力",
            passed=passed,
            severity=ERCSeverity.ERROR if not passed else ERCSeverity.INFO,
            message=f"|IOH|({self._format_current_value(ioh, ioh_unit)}) {'>=' if passed else '<'} |IIH|({self._format_current_value(iih, iih_unit)}), 余量={self._format_current_value(margin, 'A')}",
            actual_values={"IOH": ioh_amps, "IIH": iih_amps},
            expected_condition="|IOH| >= |IIH|",
            margin=margin
        )

    def _check_low_drive_capability(self, driver: Dict, receiver: Dict) -> Optional[ERCRuleResult]:
        """Check low-level drive capability [ERC-L1-I002]: IOL >= |IIL|."""
        if "IOL" not in driver or "IIL" not in receiver:
            return None

        iol = driver["IOL"]
        iil = receiver["IIL"]
        iol_unit = driver.get("IOL_unit", "mA")
        iil_unit = receiver.get("IIL_unit", "mA")

        iol_amps = self._normalize_current_to_amps(iol, iol_unit)
        iil_amps = self._normalize_current_to_amps(iil, iil_unit)

        passed = iol_amps >= abs(iil_amps)
        margin = iol_amps - abs(iil_amps)

        return ERCRuleResult(
            rule_id="ERC-L1-I002",
            rule_name="低电平驱动能力",
            passed=passed,
            severity=ERCSeverity.ERROR if not passed else ERCSeverity.INFO,
            message=f"IOL({self._format_current_value(iol, iol_unit)}) {'>=' if passed else '<'} |IIL|({self._format_current_value(iil, iil_unit)}), 余量={self._format_current_value(margin, 'A')}",
            actual_values={"IOL": iol_amps, "IIL": iil_amps},
            expected_condition="IOL >= |IIL|",
            margin=margin
        )

    def _check_fanout(self, driver: Dict, receiver: Dict) -> Optional[ERCRuleResult]:
        """Check fanout capability [ERC-L1-I003]: Fanout = min(IOL/|IIL|, |IOH|/|IIH|) >= 1."""
        ioh = driver.get("IOH")
        iol = driver.get("IOL")
        iih = receiver.get("IIH")
        iil = receiver.get("IIL")

        if None in [ioh, iol, iih, iil]:
            return None

        ioh_unit = driver.get("IOH_unit", "mA")
        iol_unit = driver.get("IOL_unit", "mA")
        iih_unit = receiver.get("IIH_unit", "mA")
        iil_unit = receiver.get("IIL_unit", "mA")

        ioh_amps = self._normalize_current_to_amps(ioh, ioh_unit)
        iol_amps = self._normalize_current_to_amps(iol, iol_unit)
        iih_amps = self._normalize_current_to_amps(iih, iih_unit)
        iil_amps = self._normalize_current_to_amps(iil, iil_unit)

        high_fanout = abs(ioh_amps) / abs(iih_amps) if abs(iih_amps) > 0 else float('inf')
        low_fanout = iol_amps / abs(iil_amps) if abs(iil_amps) > 0 else float('inf')
        fanout = min(high_fanout, low_fanout)

        passed = fanout >= 1

        return ERCRuleResult(
            rule_id="ERC-L1-I003",
            rule_name="扇出能力评估",
            passed=passed,
            severity=ERCSeverity.INFO,
            message=f"扇出={fanout:.1f} (高电平扇出={high_fanout:.1f}, 低电平扇出={low_fanout:.1f})",
            actual_values={"fanout": fanout, "high_fanout": high_fanout, "low_fanout": low_fanout},
            expected_condition="Fanout >= 1",
            margin=fanout - 1
        )

    def _check_critical_length(self, driver: Dict, _receiver: Dict,
                               trace_length: Optional[float], trace_impedance: Optional[float]) -> Optional[ERCRuleResult]:
        """Check critical transmission line length [ERC-L2-T001].

        critical_length = min(tr, tf) * signal_speed / 6
        FR4 signal speed ≈ 15 cm/ns.
        """
        rise_time = driver.get("rise_time") or driver.get("tr") or 10e-9
        fall_time = driver.get("fall_time") or driver.get("tf") or 10e-9
        signal_speed = 15

        critical_length = min(rise_time, fall_time) * 1e9 * signal_speed / 6

        if trace_length is None:
            return ERCRuleResult(
                rule_id="ERC-L2-T001",
                rule_name="临界传输线长度计算",
                passed=True,
                severity=ERCSeverity.INFO,
                message=f"临界长度={critical_length:.2f}cm (未提供走线长度)",
                actual_values={"critical_length": critical_length},
                expected_condition="trace_length < critical_length",
                margin=None
            )

        passed = trace_length < critical_length
        margin = critical_length - trace_length

        return ERCRuleResult(
            rule_id="ERC-L2-T001",
            rule_name="临界传输线长度计算",
            passed=passed,
            severity=ERCSeverity.WARNING if not passed else ERCSeverity.INFO,
            message=f"走线长度={trace_length:.2f}cm, 临界长度={critical_length:.2f}cm, {'通过' if passed else '需要端接'}",
            actual_values={"trace_length": trace_length, "critical_length": critical_length},
            expected_condition="trace_length < critical_length",
            margin=margin
        )

    def _check_reflection_coefficient(self, driver: Dict, receiver: Dict,
                                      _trace_length: Optional[float], trace_impedance: Optional[float]) -> Optional[ERCRuleResult]:
        """Check reflection coefficient [ERC-L2-R001]: |Γ| < 0.15.

        Γs = (Z0 - Zs) / (Z0 + Zs)  (source reflection)
        Γl = (Zl - Z0) / (Zl + Z0)  (load reflection)
        """
        if trace_impedance is None:
            return None

        z0 = trace_impedance
        zs = driver.get("output_impedance") or driver.get("Zo", 25.0)
        zl = receiver.get("input_impedance") or receiver.get("Zi", 1e6)

        gamma_source = abs((z0 - zs) / (z0 + zs)) if (z0 + zs) != 0 else 0
        gamma_load = abs((zl - z0) / (zl + z0)) if (zl + z0) != 0 else 0
        worst_gamma = max(gamma_source, gamma_load)

        passed = worst_gamma < 0.15
        margin = 0.15 - worst_gamma

        return ERCRuleResult(
            rule_id="ERC-L2-R001",
            rule_name="反射系数分析",
            passed=passed,
            severity=ERCSeverity.WARNING if not passed else ERCSeverity.INFO,
            message=f"源端Γ={gamma_source:.3f}, 负载端Γ={gamma_load:.3f}, 最劣|Γ|={worst_gamma:.3f}",
            actual_values={"gamma_source": gamma_source, "gamma_load": gamma_load, "worst_gamma": worst_gamma},
            expected_condition="|Γ| < 0.15",
            margin=margin
        )

    def _check_interface_contract(self, _driver: Dict, _receiver: Dict,
                                  topology_info: Dict) -> Optional[ERCRuleResult]:
        """Check interface protocol compatibility [ERC-L3-I001].

        Evaluates interface contract compatibility with three tiers:
        - Exact match (e.g. UART↔UART): fully compatible
        - Conditional match (e.g. SPI↔GPIO): requires bit-bang or level matching
        - GPIO general compatibility: one side is GPIO

        Also checks voltage domain crossing when both sides report
        their voltage domains (difference > 0.5V triggers WARNING).

        Args:
            _driver: Driver parameter dict (unused by this rule).
            _receiver: Receiver parameter dict (unused by this rule).
            topology_info: Must contain driver_interface, receiver_interface,
                and optionally driver_voltage, receiver_voltage.

        Returns:
            ERCRuleResult or None.
        """
        if topology_info is None:
            return ERCRuleResult(
                rule_id='ERC-L3-I001',
                rule_name='接口协议兼容性检查',
                passed=True,
                severity=ERCSeverity.INFO,
                message='未提供拓扑信息，假设接口契约兼容',
                actual_values={},
                expected_condition='接口契约兼容',
                margin=None,
            )

        driver_interface = topology_info.get('driver_interface', 'GPIO')
        receiver_interface = topology_info.get('receiver_interface', 'GPIO')

        exact_match = [
            ('GPIO', 'GPIO'), ('UART', 'UART'), ('I2C', 'I2C'), ('SPI', 'SPI'),
            ('ADC', 'ADC'), ('PWM', 'PWM'), ('CAN', 'CAN'),
        ]

        conditional_match = [
            ('SPI', 'GPIO'), ('GPIO', 'SPI'), ('I2C', 'GPIO'), ('GPIO', 'I2C'),
            ('UART', 'GPIO'), ('GPIO', 'UART'), ('PWM', 'GPIO'), ('GPIO', 'PWM'),
        ]

        if (driver_interface, receiver_interface) in exact_match:
            protocol_compatible = True
            protocol_note = ''
        elif (driver_interface, receiver_interface) in conditional_match:
            protocol_compatible = True
            protocol_note = f'（条件兼容：{driver_interface}↔{receiver_interface}需软件bit-bang或电平匹配）'
        elif driver_interface == 'GPIO' or receiver_interface == 'GPIO':
            protocol_compatible = True
            protocol_note = '（GPIO通用兼容）'
        else:
            protocol_compatible = False
            protocol_note = ''

        driver_voltage = topology_info.get('driver_voltage')
        receiver_voltage = topology_info.get('receiver_voltage')

        voltage_cross_warning = ''
        voltage_cross_issue = False

        if driver_voltage and receiver_voltage:
            voltage_diff = abs(driver_voltage - receiver_voltage)
            if voltage_diff > 0.5:
                voltage_cross_issue = True
                voltage_cross_warning = (
                    f'；电压域交叉风险：驱动端{driver_voltage}V ↔ '
                    f'接收端{receiver_voltage}V（差{voltage_diff:.1f}V），建议使用电平转换器'
                )

        passed = protocol_compatible and not voltage_cross_issue

        message = f'驱动端接口={driver_interface}, 接收端接口={receiver_interface}'
        if protocol_note:
            message = message + protocol_note
        if voltage_cross_warning:
            message = message + voltage_cross_warning
        if not protocol_compatible:
            message = f'驱动端接口={driver_interface}, 接收端接口={receiver_interface}, 协议不兼容'

        return ERCRuleResult(
            rule_id='ERC-L3-I001',
            rule_name='接口协议兼容性检查',
            passed=passed,
            severity=(
                ERCSeverity.ERROR if not protocol_compatible
                else (ERCSeverity.WARNING if voltage_cross_issue else ERCSeverity.INFO)
            ),
            message=message,
            actual_values={
                'driver_interface': driver_interface,
                'receiver_interface': receiver_interface,
                'driver_voltage': driver_voltage,
                'receiver_voltage': receiver_voltage,
            },
            expected_condition='接口契约兼容且电压域匹配',
            margin=None,
        )

    def _check_pam_conflict(self, _driver: Dict, _receiver: Dict,
                            topology_info: Dict) -> Optional[ERCRuleResult]:
        """Check port attribute conflicts [ERC-L3-P001].

        Detects direction conflicts, pull-up/pull-down conflicts, and
        tri-state bus conflicts based on topology information.

        Args:
            _driver: Driver parameter dict (unused by this rule).
            _receiver: Receiver parameter dict (unused by this rule).
            topology_info: Must contain driver_direction, receiver_direction,
                and optionally driver_pull, receiver_pull, driver_tri_state.

        Returns:
            ERCRuleResult or None.
        """
        if topology_info is None:
            return ERCRuleResult(
                rule_id='ERC-L3-P001',
                rule_name='端口属性冲突检测',
                passed=True,
                severity=ERCSeverity.INFO,
                message='拓扑信息不足，端口属性检查已跳过，建议人工确认端口方向与上下拉配置',
                actual_values={},
                expected_condition='端口属性无冲突',
                margin=None,
            )

        conflicts = []

        driver_direction = topology_info.get('driver_direction', 'output')
        receiver_direction = topology_info.get('receiver_direction', 'input')

        if driver_direction == 'input' and receiver_direction == 'input':
            conflicts.append('驱动端和接收端均为输入方向，无有效驱动源')

        if driver_direction == 'bidirectional' and receiver_direction == 'bidirectional':
            conflicts.append('两端均为双向端口，存在总线竞争风险')

        driver_pull = topology_info.get('driver_pull')
        receiver_pull = topology_info.get('receiver_pull')

        if driver_pull == 'pull_up' and receiver_pull == 'pull_down':
            conflicts.append('驱动端上拉与接收端下拉存在直流通路冲突')

        if driver_pull == 'pull_down' and receiver_pull == 'pull_up':
            conflicts.append('驱动端下拉与接收端上拉存在直流通路冲突')

        driver_tri_state = topology_info.get('driver_tri_state', False)

        if driver_tri_state and receiver_direction != 'input':
            conflicts.append('驱动端为三态输出但接收端非输入方向，可能存在总线冲突')

        passed = len(conflicts) == 0

        return ERCRuleResult(
            rule_id='ERC-L3-P001',
            rule_name='端口属性冲突检测',
            passed=passed,
            severity=ERCSeverity.ERROR if not passed else ERCSeverity.INFO,
            message='端口属性无冲突' if passed else '; '.join(conflicts),
            actual_values={
                'driver_direction': driver_direction,
                'receiver_direction': receiver_direction,
                'driver_pull': driver_pull,
                'receiver_pull': receiver_pull,
            },
            expected_condition='端口属性无冲突',
            margin=None,
        )

    def _check_temperature_drift(self, driver: Dict, receiver: Dict, temperature: float) -> Optional[ERCRuleResult]:
        """Check temperature drift using interval arithmetic [ERC-L4-T001].

        Expands electrical parameters into intervals based on temperature
        drift, then checks whether noise margins remain positive.
        """
        voh = driver.get("VOH")
        vol = driver.get("VOL")
        vih = receiver.get("VIH")
        vil = receiver.get("VIL")

        if None in [voh, vol, vih, vil]:
            return None

        temp_drift_coeff = 0.002
        delta_t = temperature - 25.0
        drift = abs(delta_t) * temp_drift_coeff

        voh_interval = Interval.from_crisp_with_drift(voh, voh * drift)
        vol_interval = Interval.from_crisp_with_drift(vol, vol * drift)
        vih_interval = Interval.from_crisp_with_drift(vih, vih * drift)
        vil_interval = Interval.from_crisp_with_drift(vil, vil * drift)

        high_margin = voh_interval - vih_interval
        low_margin = vil_interval - vol_interval

        if high_margin.definitely_positive() and low_margin.definitely_positive():
            passed = True
            severity = ERCSeverity.INFO
        elif high_margin.possibly_negative() or low_margin.possibly_negative():
            passed = False
            severity = ERCSeverity.WARNING
        else:
            passed = True
            severity = ERCSeverity.WARNING

        return ERCRuleResult(
            rule_id="ERC-L4-T001",
            rule_name="温度漂移检测",
            passed=passed,
            severity=severity,
            message=f"温度={temperature}°C, 高电平容限=[{high_margin.lo:.3f}, {high_margin.hi:.3f}]V, 低电平容限=[{low_margin.lo:.3f}, {low_margin.hi:.3f}]V",
            actual_values={
                "temperature": temperature,
                "high_margin_lo": high_margin.lo, "high_margin_hi": high_margin.hi,
                "low_margin_lo": low_margin.lo, "low_margin_hi": low_margin.hi,
            },
            expected_condition="温度漂移后噪声容限为正",
            margin=min(high_margin.lo, low_margin.lo)
        )

    def _check_process_degradation(self, driver: Dict, receiver: Dict, temperature: float) -> Optional[ERCRuleResult]:
        """Check process degradation using Arrhenius model [ERC-L4-P001]."""
        driver_process, driver_package, driver_voltage = self._parse_chip_params(driver)
        receiver_process, receiver_package, receiver_voltage = self._parse_chip_params(receiver)

        driver_result = self.degradation_db.calculate_degradation_factor(
            driver_process, driver_package,
            temperature=temperature,
            voltage=driver_voltage
        )
        receiver_result = self.degradation_db.calculate_degradation_factor(
            receiver_process, receiver_package,
            temperature=temperature,
            voltage=receiver_voltage
        )

        worst_result = driver_result if driver_result.degradation_factor < receiver_result.degradation_factor else receiver_result
        degradation_factor = worst_result.degradation_factor

        voh = driver.get("VOH", 3.3)
        vol = driver.get("VOL", 0.1)
        vih = receiver.get("VIH", 2.0)
        vil = receiver.get("VIL", 0.8)

        voh_degraded = voh * degradation_factor
        vol_degraded = vol * (2.0 - degradation_factor)

        high_margin = voh_degraded - vih
        low_margin = vil - vol_degraded

        passed = high_margin >= self.MIN_MARGIN_THRESHOLD and low_margin >= self.MIN_MARGIN_THRESHOLD

        high_margin_fresh = voh - vih
        low_margin_fresh = vil - vol

        message = (
            f"退化因子={degradation_factor:.3f} (置信区间: [{worst_result.confidence_interval[0]:.3f}, "
            f"{worst_result.confidence_interval[1]:.3f}]), "
            f"高电平余量={high_margin:.3f}V(原始{high_margin_fresh:.3f}V), "
            f"低电平余量={low_margin:.3f}V(原始{low_margin_fresh:.3f}V), "
            f"风险等级={worst_result.risk_level}, {'通过' if passed else '风险'}"
        )

        return ERCRuleResult(
            rule_id="ERC-L4-P001",
            rule_name="工艺退化基准库查询",
            passed=passed,
            severity=ERCSeverity.WARNING if not passed else ERCSeverity.INFO,
            message=message,
            actual_values={
                "degradation_factor": degradation_factor,
                "high_margin_fresh": high_margin_fresh,
                "low_margin_fresh": low_margin_fresh,
                "high_margin": high_margin,
                "low_margin": low_margin,
                "confidence_interval": worst_result.confidence_interval,
                "risk_level": worst_result.risk_level,
                "estimated_lifetime": worst_result.estimated_lifetime,
                "driver_process": driver_process.value,
                "driver_package": driver_package.value,
                "receiver_process": receiver_process.value,
                "receiver_package": receiver_package.value
            },
            expected_condition="工艺退化在可接受范围内",
            margin=min(high_margin, low_margin) - self.MIN_MARGIN_THRESHOLD,
            suggestion="; ".join(worst_result.recommendations) if worst_result.recommendations else ""
        )

    def _parse_chip_params(self, chip_params: Dict) -> Tuple[ProcessType, PackageType, float]:
        """Extract process type, package type, and voltage from chip parameters."""
        process = self.degradation_db.parse_process_type(
            chip_params.get("process", chip_params.get("family", "CMOS"))
        )
        package = self.degradation_db.parse_package_type(
            chip_params.get("package", "DIP")
        )
        voltage = chip_params.get("VCC", chip_params.get("supply_voltage", 3.3))
        return process, package, voltage

    def _evaluate_overall_compatibility(self, result: FourLayerERCResult) -> bool:
        """Evaluate overall compatibility.

        Layer 1 (static stability) and Layer 3 (topology conflict) are
        hard requirements. Layer 2 and Layer 4 failures are advisory only.
        """
        if not result.layer1_result or not result.layer1_result.passed:
            return False
        if not result.layer3_result or not result.layer3_result.passed:
            return False
        return True

    def _has_warnings(self, result: FourLayerERCResult) -> bool:
        """Check whether Layer 2 or Layer 4 has warnings."""
        if result.layer2_result and not result.layer2_result.passed:
            return True
        if result.layer4_result and not result.layer4_result.passed:
            return True
        return False

    def _calculate_overall_confidence(self, result: FourLayerERCResult) -> float:
        """Calculate overall confidence as passed_rules / total_rules."""
        total_rules = 0
        passed_rules = 0

        layer_results = [result.layer1_result, result.layer2_result,
                         result.layer3_result, result.layer4_result]

        for layer_result in layer_results:
            if layer_result:
                total_rules += len(layer_result.results)
                passed_rules += sum(1 for r in layer_result.results if r.passed)

        return passed_rules / total_rules if total_rules > 0 else 0.0

    def _generate_layer_summary(self, layer_num: int, passed: bool,
                                 results: List[ERCRuleResult],
                                 driver_chip: str, receiver_chip: str) -> str:
        """Generate a human-readable summary for a single layer."""
        layer_config = {
            1: {"name": "静态稳定性检测", "success_msg": "电气参数兼容",
                "fail_msg": "电气参数不兼容", "severity": ERCSeverity.ERROR, "icon": "❌"},
            2: {"name": "准物理反射预警", "success_msg": "信号完整性良好",
                "fail_msg": "信号完整性风险", "severity": ERCSeverity.WARNING, "icon": "⚠️"},
            3: {"name": "拓扑冲突仲裁", "success_msg": "接口契约兼容",
                "fail_msg": "接口契约冲突", "severity": ERCSeverity.ERROR, "icon": "❌"},
            4: {"name": "极端环境退化评估", "success_msg": "在极端环境下稳定",
                "fail_msg": "在极端环境下存在风险", "severity": ERCSeverity.WARNING, "icon": "⚠️"},
        }

        config = layer_config[layer_num]

        if passed:
            return f"✅ {config['name']}通过：{driver_chip}与{receiver_chip}{config['success_msg']}"

        issues = [r for r in results if r.severity == config['severity'] and not r.passed]
        issue_msgs = [f"- {r.message}" for r in issues]
        return (f"{config['icon']} {config['name']}失败：{driver_chip}与{receiver_chip}"
                f"存在{config['fail_msg']}\n" + "\n".join(issue_msgs))

    def _generate_overall_summary(self, result: FourLayerERCResult) -> str:
        """Generate overall summary."""
        if result.overall_compatible:
            if self._has_warnings(result):
                warning_layers = []
                if result.layer2_result and not result.layer2_result.passed:
                    warning_layers.append("信号完整性风险")
                if result.layer4_result and not result.layer4_result.passed:
                    warning_layers.append("极端环境风险")
                return (f"⚠️ 四层ERC检查：基本兼容但存在警告"
                        f"（{', '.join(warning_layers)}），置信度={result.overall_confidence:.2%}")
            return f"✅ 四层ERC检查通过：完全兼容，置信度={result.overall_confidence:.2%}"

        issues = []
        if result.layer1_result and not result.layer1_result.passed:
            issues.append("静态稳定性检测失败")
        if result.layer3_result and not result.layer3_result.passed:
            issues.append("拓扑冲突仲裁失败")
        return f"❌ 四层ERC检查失败：{', '.join(issues)}"

    def _generate_suggestions(self, result: FourLayerERCResult) -> List[str]:
        """Generate improvement suggestions based on check results."""
        suggestions = []

        suggestion_map = {
            "layer1": "建议检查驱动端与接收端的电气参数，确保VOH>=VIH且VOL<=VIL",
            "layer2": "建议添加端接电阻以改善信号完整性",
            "layer3": "建议检查接口契约，确保驱动端与接收端接口类型匹配",
            "layer4": "建议在极端环境下进行额外测试，确保系统稳定性"
        }

        layer_results = [
            (result.layer1_result, "layer1"),
            (result.layer2_result, "layer2"),
            (result.layer3_result, "layer3"),
            (result.layer4_result, "layer4")
        ]

        for layer_result, layer_key in layer_results:
            if layer_result and not layer_result.passed:
                suggestions.append(suggestion_map[layer_key])

        if not suggestions:
            suggestions.append("系统设计良好，建议继续进行其他验证")

        return suggestions
