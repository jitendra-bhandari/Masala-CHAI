"""
Agents V2: Refinement-Based Pipeline with Training Data Collection

Architecture:
- Agent 1: Component Refinement (VLM)
- Agent 2: Connectivity Refinement (VLM)  
- Agent 3: Netlist Generation (LLM)
- LLM-as-Judge: Cross-Validation (Mandatory)

Features:
- ReAct-style agents with error context propagation
- Graceful continuation on failure with NEEDS_REVIEW flags
- Comprehensive training data collection for curriculum learning
- Multi-head YOLO training data export
"""

from .workflow import run_pipeline, RefinementPipeline
from .config import PipelineConfig, ModelConfig

__version__ = "2.0.0"
__all__ = ["run_pipeline", "RefinementPipeline", "PipelineConfig", "ModelConfig"]

