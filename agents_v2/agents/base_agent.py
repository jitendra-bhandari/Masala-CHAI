"""
Base ReAct Agent with Error Context Propagation

This implements the core ReAct (Reason + Act) loop with:
1. Error context from previous iterations
2. Metric-based evaluation
3. Graceful continuation on failure
4. Training data collection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import time
import json
import base64
from io import BytesIO
from PIL import Image

import os
from openai import OpenAI

from ..config import PipelineConfig, QualityTier
from ..models.output import StageResult, IterationRecord
from ..models.training_data import RefinementDeltas, ChangeLogEntry


@dataclass
class AgentContext:
    """Context passed between iterations"""
    
    # Input data
    original_input: Dict[str, Any] = field(default_factory=dict)
    
    # Previous iteration output
    previous_output: Optional[Dict] = None
    
    # Metrics from previous iteration
    previous_metrics: Dict[str, float] = field(default_factory=dict)
    passed_metrics: List[str] = field(default_factory=list)
    failed_metrics: List[str] = field(default_factory=list)
    
    # Specific errors to fix
    errors: List[str] = field(default_factory=list)
    
    # Actionable feedback
    feedback: str = ""
    
    # Upstream quality flags
    upstream_needs_review: bool = False
    upstream_confidence: float = 1.0


class BaseReActAgent(ABC):
    """
    Base class for ReAct-style agents with error context propagation.
    
    Each iteration:
    1. THINK: Analyze input + previous errors
    2. ACT: Perform the task
    3. OBSERVE: Evaluate metrics
    4. REFLECT: If failed, build error context for next iteration
    """
    
    def __init__(
        self,
        agent_name: str,
        model: str,
        config: PipelineConfig,
        requires_vision: bool = True
    ):
        self.agent_name = agent_name
        self.model = model
        self.config = config
        self.requires_vision = requires_vision
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable.\n"
                "Example: export OPENAI_API_KEY='your-api-key'"
            )
        self.client = OpenAI(api_key=api_key)
        
        # Iteration tracking
        self.current_iteration = 0
        self.iteration_history: List[IterationRecord] = []
        self.best_output: Optional[Dict] = None
        self.best_score: float = 0.0
        self.best_iteration: int = 0
        
        # Refinement tracking
        self.refinement_deltas = RefinementDeltas()
        
        # Token tracking
        self.total_tokens = 0
    
    @abstractmethod
    def get_max_iterations(self) -> int:
        """Get max iterations for this agent"""
        pass
    
    @abstractmethod
    def get_pass_threshold(self) -> float:
        """Get pass threshold for this agent"""
        pass
    
    @abstractmethod
    def get_min_viable_threshold(self) -> float:
        """Get minimum viable threshold"""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass
    
    @abstractmethod
    def build_user_prompt(self, context: AgentContext, iteration: int) -> str:
        """Build the user prompt including error context"""
        pass
    
    @abstractmethod
    def parse_response(self, response: str, context: AgentContext) -> Dict[str, Any]:
        """Parse the LLM response into structured output"""
        pass
    
    @abstractmethod
    def evaluate_metrics(self, output: Dict[str, Any], context: AgentContext) -> Dict[str, float]:
        """Evaluate the output and return metrics"""
        pass
    
    @abstractmethod
    def build_error_context(
        self,
        output: Dict[str, Any],
        metrics: Dict[str, float],
        context: AgentContext
    ) -> AgentContext:
        """Build error context for next iteration"""
        pass
    
    def run(
        self,
        input_data: Dict[str, Any],
        images: Optional[List[Image.Image]] = None,
        upstream_context: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], StageResult]:
        """
        Run the agent with ReAct loop.
        
        Returns:
            Tuple of (best_output, stage_result)
        """
        start_time = time.time()
        
        # Initialize context
        context = AgentContext(
            original_input=input_data,
            upstream_needs_review=upstream_context.get("needs_review", False) if upstream_context else False,
            upstream_confidence=upstream_context.get("confidence", 1.0) if upstream_context else 1.0
        )
        
        # Initialize stage result
        stage_result = StageResult(
            stage_name=self.agent_name,
            model_used=self.model
        )
        
        max_iterations = self.get_max_iterations()
        
        for iteration in range(1, max_iterations + 1):
            self.current_iteration = iteration
            iter_start = time.time()
            
            if self.config.verbose:
                print(f"\n{'='*60}")
                print(f"  {self.agent_name} - Iteration {iteration}/{max_iterations}")
                print(f"{'='*60}")
            
            # Build prompt with error context
            user_prompt = self.build_user_prompt(context, iteration)
            
            # Call LLM
            try:
                response, tokens = self._call_llm(
                    self.get_system_prompt(),
                    user_prompt,
                    images if self.requires_vision else None
                )
                self.total_tokens += tokens
            except Exception as e:
                print(f"  ERROR: LLM call failed: {e}")
                context.errors.append(f"LLM call failed: {str(e)}")
                continue
            
            # Parse response
            try:
                output = self.parse_response(response, context)
            except Exception as e:
                print(f"  ERROR: Failed to parse response: {e}")
                context.errors.append(f"Parse failed: {str(e)}")
                continue
            
            # Evaluate metrics
            metrics = self.evaluate_metrics(output, context)
            score = self._compute_score(metrics)
            
            # Check which metrics passed/failed
            passed_metrics = []
            failed_metrics = []
            for metric, value in metrics.items():
                target = self._get_metric_target(metric)
                if value >= target:
                    passed_metrics.append(metric)
                else:
                    failed_metrics.append(metric)
            
            # Record iteration
            iter_record = IterationRecord(
                iteration_number=iteration,
                metrics=metrics,
                score=score,
                passed=score >= self.get_pass_threshold(),
                actions_taken=output.get("actions", []),
                errors_found=failed_metrics,
                duration_seconds=time.time() - iter_start,
                tokens_used=tokens
            )
            self.iteration_history.append(iter_record)
            stage_result.add_iteration(iter_record)
            
            if self.config.verbose:
                print(f"\n  Metrics:")
                for m, v in metrics.items():
                    status = "✓" if m in passed_metrics else "✗"
                    print(f"    {m}: {v:.3f} {status}")
                print(f"\n  Score: {score:.3f} (target: {self.get_pass_threshold():.2f})")
            
            # Update best if improved
            if score > self.best_score:
                self.best_score = score
                self.best_output = output
                self.best_iteration = iteration
                if self.config.verbose:
                    print(f"  ✓ New best score!")
            
            # Check if passed
            if score >= self.get_pass_threshold():
                if self.config.verbose:
                    print(f"\n  ✓ PASSED - All metrics satisfied")
                break
            
            # Build error context for next iteration
            if iteration < max_iterations:
                context = self.build_error_context(output, metrics, context)
                context.previous_output = output
                context.previous_metrics = metrics
                context.passed_metrics = passed_metrics
                context.failed_metrics = failed_metrics
                
                if self.config.verbose:
                    print(f"\n  ✗ FAILED - Preparing error context for next iteration")
                    print(f"  Failed metrics: {failed_metrics}")
        
        # Finalize stage result
        stage_result.final_metrics = metrics if 'metrics' in dir() else {}
        stage_result.final_score = self.best_score
        stage_result.passed = self.best_score >= self.get_pass_threshold()
        stage_result.best_output = self.best_output
        stage_result.total_duration = time.time() - start_time
        stage_result.total_tokens = self.total_tokens
        
        # Determine if needs review
        if self.best_score < self.get_min_viable_threshold():
            stage_result.needs_review = True
            stage_result.review_reasons.append(
                f"Score {self.best_score:.2f} below minimum viable {self.get_min_viable_threshold()}"
            )
        elif self.best_score < self.get_pass_threshold():
            stage_result.needs_review = True
            stage_result.review_reasons.append(
                f"Score {self.best_score:.2f} below pass threshold {self.get_pass_threshold()}"
            )
        
        if self.config.verbose:
            print(f"\n  {'='*50}")
            print(f"  {self.agent_name} COMPLETE")
            print(f"  Best Score: {self.best_score:.3f} (iteration {self.best_iteration})")
            print(f"  Passed: {stage_result.passed}")
            print(f"  Needs Review: {stage_result.needs_review}")
            print(f"  {'='*50}")
        
        return self.best_output or {}, stage_result
    
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[List[Image.Image]] = None
    ) -> Tuple[str, int]:
        """Call the LLM with optional images"""
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Build user message with images
        if images and self.requires_vision:
            content = [{"type": "text", "text": user_prompt}]
            for img in images:
                # Cap at 1024px longest side — sufficient for vision, avoids token explosion
                _img = img.copy()
                if max(_img.size) > 1024:
                    _img.thumbnail((1024, 1024))
                # Convert to RGB for JPEG (drops alpha channel if present)
                if _img.mode in ("RGBA", "P"):
                    _img = _img.convert("RGB")
                buffered = BytesIO()
                _img.save(buffered, format="JPEG", quality=85)
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": "high"
                    }
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_prompt})
        
        # Call API - handle different parameter names for different models
        # Newer models (o1, gpt-5.x) use max_completion_tokens
        # Older models (gpt-4o, gpt-4) use max_tokens
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2
        }
        
        # Check if model uses new API format
        if any(x in self.model.lower() for x in ["o1", "o3", "o4", "gpt-5", "5.2"]):
            api_params["max_completion_tokens"] = 4096
        else:
            api_params["max_tokens"] = 4096
        
        response = self.client.chat.completions.create(**api_params)
        
        tokens = response.usage.total_tokens if response.usage else 0
        content = response.choices[0].message.content or ""
        
        return content, tokens
    
    def _compute_score(self, metrics: Dict[str, float]) -> float:
        """Compute weighted score from metrics"""
        weights = self._get_metric_weights()
        total_weight = sum(weights.get(m, 1.0) for m in metrics)
        score = sum(
            metrics[m] * weights.get(m, 1.0)
            for m in metrics
        ) / total_weight if total_weight > 0 else 0.0
        return score
    
    def _get_metric_weights(self) -> Dict[str, float]:
        """Get metric weights - override in subclass"""
        return {}
    
    def _get_metric_target(self, metric: str) -> float:
        """Get target value for a metric - override in subclass"""
        return 0.8
    
    def log_change(
        self,
        action: str,
        target_type: str,
        target_id: str,
        **kwargs
    ):
        """Log a refinement change"""
        self.refinement_deltas.add_change(
            agent=self.agent_name,
            iteration=self.current_iteration,
            action=action,
            target_type=target_type,
            target_id=target_id,
            **kwargs
        )

