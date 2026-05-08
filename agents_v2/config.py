"""
Configuration for Agents V2 Pipeline
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class QualityTier(Enum):
    """Output quality classification"""
    VERIFIED = "verified"           # Score >= 0.90, all checks passed
    CONFIDENT = "confident"         # Score 0.80-0.90, minor issues
    LOW_CONFIDENCE = "low_confidence"  # Score 0.60-0.80, proceed with caution
    FAILED = "failed"               # Score < 0.60, needs review
    PARTIAL = "partial"             # Some stages succeeded, others failed


class PipelineStatus(Enum):
    """Overall pipeline status"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


@dataclass
class ModelConfig:
    """Model configuration for each agent"""
    
    # Agent models 
    # NOTE: GPT-5.2 models may have different API requirements
    # Set USE_GPT5 = True when ready to use GPT-5.2
    # For now, default to gpt-4o which is known to work
    
    # Production config (when GPT-5.2 is ready):
    # agent1_model: str = "gpt-5.2"
    # agent2_model: str = "gpt-5.2"
    # agent3_model: str = "gpt-5.2"
    # judge_model: str = "gpt-5.2-pro"
    
    # Production config with GPT-5.2:
    agent1_model: str = "gpt-5.2"           # Component Refinement
    agent2_model: str = "gpt-5.2"           # Connectivity Refinement
    agent3_model: str = "gpt-5.2"           # Netlist Generation
    judge_model: str = "gpt-5.2"            # LLM-as-Judge (mandatory) - same as agents
    
    # Fallback models (if primary fails)
    fallback_model: str = "gpt-4o-mini"
    
    # API configuration
    api_timeout: int = 60
    max_retries: int = 3
    
    # Flag to switch to GPT-5.2 when available
    use_gpt5: bool = False
    
    def get_agent_model(self, agent_num: int) -> str:
        """Get model for agent, considering GPT-5 flag"""
        if self.use_gpt5:
            if agent_num <= 3:
                return "gpt-5.2"
            else:
                return "gpt-5.2-pro"
        else:
            models = {1: self.agent1_model, 2: self.agent2_model, 
                     3: self.agent3_model, 4: self.judge_model}
            return models.get(agent_num, self.agent1_model)


@dataclass
class ThresholdConfig:
    """Threshold configuration for each stage"""
    
    # Agent 1: Component Refinement
    agent1_pass_threshold: float = 0.90
    agent1_acceptable_threshold: float = 0.70
    agent1_min_viable_threshold: float = 0.50
    
    # Agent 2: Connectivity Refinement
    agent2_pass_threshold: float = 0.90
    agent2_acceptable_threshold: float = 0.70
    agent2_min_viable_threshold: float = 0.50
    
    # Agent 3: Netlist Generation
    agent3_pass_threshold: float = 0.95
    agent3_acceptable_threshold: float = 0.75
    agent3_min_viable_threshold: float = 0.50
    
    # LLM-as-Judge
    judge_pass_threshold: float = 0.96
    judge_acceptable_threshold: float = 0.70
    judge_min_confidence: float = 0.60


@dataclass
class IterationConfig:
    """Iteration limits for each agent"""
    
    agent1_max_iterations: int = 3
    agent2_max_iterations: int = 3
    agent3_max_iterations: int = 3
    judge_max_retries: int = 2
    
    # Early stopping
    min_improvement_threshold: float = 0.02  # Stop if improvement < 2%
    stuck_iterations_limit: int = 2  # Stop if no improvement for N iterations


@dataclass
class MetricWeights:
    """Weights for combining metrics into scores"""
    
    # Agent 1 metric weights
    agent1_weights: Dict[str, float] = field(default_factory=lambda: {
        "refinement_activity": 0.10,
        "component_coverage": 0.15,
        "terminal_completeness": 0.35,
        "type_confidence": 0.20,
        "visual_grounding": 0.20
    })
    
    # Agent 2 metric weights
    agent2_weights: Dict[str, float] = field(default_factory=lambda: {
        "wire_filtering": 0.15,
        "connection_completeness": 0.30,
        "node_validity": 0.20,
        "electrical_validity": 0.20,
        "visual_grounding": 0.15
    })
    
    # Agent 3 metric weights
    agent3_weights: Dict[str, float] = field(default_factory=lambda: {
        "syntax_valid": 0.25,
        "simulation_runs": 0.30,
        "topology_preserved": 0.25,
        "component_match": 0.10,
        "node_match": 0.10
    })


@dataclass
class DifficultyConfig:
    """Configuration for difficulty scoring"""
    
    # EMA decay factor for curriculum learning
    ema_decay: float = 0.9
    
    # Difficulty factor weights
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        "component_count": 0.10,
        "component_density": 0.10,
        "wire_density": 0.10,
        "crossing_count": 0.15,
        "yolo_miss_rate": 0.15,
        "yolo_fp_rate": 0.10,
        "agent_iterations": 0.15,
        "image_quality": 0.05,
        "line_noise_ratio": 0.10
    })
    
    # Curriculum stage thresholds
    stage_thresholds: List[float] = field(default_factory=lambda: [
        0.2,  # Stage 1: easiest (difficulty < 0.2)
        0.4,  # Stage 2: easy
        0.6,  # Stage 3: medium
        0.8,  # Stage 4: hard
        1.0   # Stage 5: hardest
    ])


@dataclass
class ExportConfig:
    """Configuration for training data export"""
    
    # Export formats
    export_yolo: bool = True
    export_coco: bool = True
    export_custom: bool = True
    
    # YOLO class mapping
    component_classes: Dict[str, int] = field(default_factory=lambda: {
        "Resistor": 0,
        "Capacitor": 1,
        "Inductor": 2,
        "NMOS": 3,
        "PMOS": 4,
        "NPN": 5,
        "PNP": 6,
        "Diode": 7,
        "VoltageSource": 8,
        "CurrentSource": 9,
        "Ground": 10,
        "OpAmp": 11,
        "Other": 12
    })
    
    # Wire segment classes
    wire_classes: Dict[str, int] = field(default_factory=lambda: {
        "wire": 0,
        "component_internal": 1,
        "noise": 2,
        "text": 3
    })
    
    # Node classes
    node_classes: Dict[str, int] = field(default_factory=lambda: {
        "junction": 0,
        "crossing": 1,
        "terminal": 2,
        "corner": 3
    })


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    
    models: ModelConfig = field(default_factory=ModelConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    iterations: IterationConfig = field(default_factory=IterationConfig)
    metric_weights: MetricWeights = field(default_factory=MetricWeights)
    difficulty: DifficultyConfig = field(default_factory=DifficultyConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Pipeline behavior
    continue_on_failure: bool = True  # Always continue with best output
    collect_training_data: bool = True
    save_intermediate_outputs: bool = True
    verbose: bool = True
    
    # Output paths
    output_dir: str = "agents_v2_output"
    training_data_dir: str = "training_data"


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()

