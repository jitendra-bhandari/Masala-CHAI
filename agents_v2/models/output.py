"""
Pipeline output models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

from .components import ComponentInfo
from .wires import WireInfo
from .nodes import NodeInfo, TopologyAnnotation
from .training_data import TrainingDataSample, RefinementDeltas
from .difficulty import DifficultyScores


@dataclass
class IterationRecord:
    """Record of a single iteration"""
    
    iteration_number: int
    metrics: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    passed: bool = False
    
    # What changed
    actions_taken: List[str] = field(default_factory=list)
    errors_found: List[str] = field(default_factory=list)
    feedback_given: str = ""
    
    # Context passed to next iteration
    error_context: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    duration_seconds: float = 0.0
    tokens_used: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "iteration_number": self.iteration_number,
            "metrics": self.metrics,
            "score": self.score,
            "passed": self.passed,
            "actions_taken": self.actions_taken,
            "errors_found": self.errors_found,
            "feedback_given": self.feedback_given,
            "error_context": self.error_context,
            "duration_seconds": self.duration_seconds,
            "tokens_used": self.tokens_used
        }


@dataclass
class StageResult:
    """Result from each pipeline stage"""
    
    stage_name: str
    model_used: str
    iterations: int = 0
    
    # Per-iteration history
    iteration_history: List[IterationRecord] = field(default_factory=list)
    
    # Final metrics
    final_metrics: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    passed: bool = False
    
    # Best iteration (may not be last)
    best_iteration: int = 0
    best_score: float = 0.0
    best_output: Optional[Dict] = None
    
    # Quality flags
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    
    # Timing
    total_duration: float = 0.0
    total_tokens: int = 0
    
    def add_iteration(self, record: IterationRecord):
        """Add an iteration record and update best"""
        self.iteration_history.append(record)
        self.iterations = len(self.iteration_history)
        
        if record.score > self.best_score:
            self.best_score = record.score
            self.best_iteration = record.iteration_number
    
    def to_dict(self) -> Dict:
        return {
            "stage_name": self.stage_name,
            "model_used": self.model_used,
            "iterations": self.iterations,
            "iteration_history": [r.to_dict() for r in self.iteration_history],
            "final_metrics": self.final_metrics,
            "final_score": self.final_score,
            "passed": self.passed,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "total_duration": self.total_duration,
            "total_tokens": self.total_tokens
        }


@dataclass
class RawCVOutput:
    """Preserved raw CV detection output"""
    
    # YOLO
    yolo_detections: List[Dict] = field(default_factory=list)
    yolo_confidence_threshold: float = 0.25
    yolo_model: str = ""
    
    # Hough
    hough_lines: List[Dict] = field(default_factory=list)  # {"start": [x,y], "end": [x,y]}
    hough_threshold: float = 0.7
    
    # Junctions
    junction_candidates: List[Dict] = field(default_factory=list)  # {"x": x, "y": y}
    
    # Image info
    image_width: int = 0
    image_height: int = 0
    image_hash: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "yolo_detections": self.yolo_detections,
            "yolo_confidence_threshold": self.yolo_confidence_threshold,
            "yolo_model": self.yolo_model,
            "hough_lines": self.hough_lines,
            "hough_threshold": self.hough_threshold,
            "junction_candidates": self.junction_candidates,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_hash": self.image_hash
        }


@dataclass
class NetlistOutput:
    """Generated netlist with metadata"""
    
    spice_code: str = ""              # Raw SPICE netlist content
    spice_file_path: str = ""         # Path to saved .sp file
    python_generator: str = ""        # Python code that generates it
    
    # Parsed components (for verification)
    netlist_components: List[Dict] = field(default_factory=list)
    netlist_nodes: List[str] = field(default_factory=list)
    
    # Simulation info
    simulation_attempted: bool = False
    simulation_successful: bool = False
    simulation_logs: str = ""
    simulation_time: float = 0.0
    simulation_results: Optional[Dict] = None  # Node voltages, etc.
    
    def to_dict(self) -> Dict:
        return {
            "spice_code": self.spice_code,
            "spice_file_path": self.spice_file_path,
            "python_generator": self.python_generator,
            "netlist_components": self.netlist_components,
            "netlist_nodes": self.netlist_nodes,
            "simulation_attempted": self.simulation_attempted,
            "simulation_successful": self.simulation_successful,
            "simulation_logs": self.simulation_logs,
            "simulation_time": self.simulation_time,
            "simulation_results": self.simulation_results
        }


@dataclass
class PipelineOutput:
    """Complete output with all spatial and topological information"""
    
    # ============= METADATA =============
    image_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_version: str = "2.0.0"
    
    # ============= QUALITY FLAGS =============
    overall_status: str = "pending"   # SUCCESS | PARTIAL | FAILED
    quality_tier: str = "pending"     # VERIFIED | CONFIDENT | LOW_CONFIDENCE | FAILED
    overall_confidence: float = 0.0
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # ============= STAGE RESULTS =============
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    
    # ============= COMPONENTS (with coordinates) =============
    components: List[ComponentInfo] = field(default_factory=list)
    
    # ============= WIRES (with coordinates) =============
    wires: List[WireInfo] = field(default_factory=list)
    
    # ============= NODES (with positions) =============
    nodes: List[NodeInfo] = field(default_factory=list)
    
    # ============= TOPOLOGY GRAPH =============
    topology: Optional[TopologyAnnotation] = None
    
    # ============= NETLIST =============
    netlist: Optional[NetlistOutput] = None
    
    # ============= RAW CV DATA (preserved) =============
    raw_cv_output: Optional[RawCVOutput] = None
    
    # ============= REFINEMENT TRACKING =============
    refinement_deltas: Optional[RefinementDeltas] = None
    
    # ============= DIFFICULTY SCORES =============
    difficulty_scores: Optional[DifficultyScores] = None
    
    # ============= TRAINING DATA =============
    training_sample: Optional[TrainingDataSample] = None
    
    # ============= TIMING =============
    total_time: float = 0.0
    total_llm_calls: int = 0
    total_tokens: int = 0
    
    def get_stage_result(self, stage: str) -> Optional[StageResult]:
        return self.stage_results.get(stage)
    
    def set_stage_result(self, stage: str, result: StageResult):
        self.stage_results[stage] = result
        
        # Update needs_review if any stage needs review
        if result.needs_review:
            self.needs_review = True
            self.review_reasons.extend(result.review_reasons)
    
    def compute_overall_status(self):
        """Compute overall status from stage results"""
        if not self.stage_results:
            self.overall_status = "pending"
            return
        
        all_passed = all(r.passed for r in self.stage_results.values())
        any_passed = any(r.passed for r in self.stage_results.values())
        
        if all_passed:
            self.overall_status = "SUCCESS"
            if self.overall_confidence >= 0.90:
                self.quality_tier = "VERIFIED"
            elif self.overall_confidence >= 0.80:
                self.quality_tier = "CONFIDENT"
            else:
                self.quality_tier = "LOW_CONFIDENCE"
        elif any_passed:
            self.overall_status = "PARTIAL"
            self.quality_tier = "LOW_CONFIDENCE"
            self.needs_review = True
        else:
            self.overall_status = "FAILED"
            self.quality_tier = "FAILED"
            self.needs_review = True
    
    def to_dict(self) -> Dict:
        return {
            "image_path": self.image_path,
            "timestamp": self.timestamp,
            "pipeline_version": self.pipeline_version,
            "overall_status": self.overall_status,
            "quality_tier": self.quality_tier,
            "overall_confidence": self.overall_confidence,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "warnings": self.warnings,
            "errors": self.errors,
            "stage_results": {k: v.to_dict() for k, v in self.stage_results.items()},
            "components": [c.to_dict() for c in self.components],
            "wires": [w.to_dict() for w in self.wires],
            "nodes": [n.to_dict() for n in self.nodes],
            "topology": self.topology.to_dict() if self.topology else None,
            "netlist": self.netlist.to_dict() if self.netlist else None,
            "raw_cv_output": self.raw_cv_output.to_dict() if self.raw_cv_output else None,
            "refinement_deltas": self.refinement_deltas.to_dict() if self.refinement_deltas else None,
            "difficulty_scores": self.difficulty_scores.to_dict() if self.difficulty_scores else None,
            "training_sample": self.training_sample.to_dict() if self.training_sample else None,
            "total_time": self.total_time,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens
        }
    
    def save(self, output_path: str):
        """Save output to JSON file"""
        import json
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, input_path: str) -> 'PipelineOutput':
        """Load output from JSON file"""
        import json
        with open(input_path, 'r') as f:
            data = json.load(f)
        # Would need full deserialization implementation
        return cls(image_path=data.get("image_path", ""))

