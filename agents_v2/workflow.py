"""
Refinement Pipeline Workflow

Orchestrates the multi-agent pipeline for circuit schematic analysis:
1. Component Refinement (validate YOLO detections)
2. Connectivity Refinement (validate Hough lines)
3. Netlist Generation (create SPICE)
4. LLM-as-Judge (validate)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import hashlib

from .agents.component_agent import ComponentRefinementAgent
from .agents.connectivity_agent import ConnectivityRefinementAgent
from .agents.netlist_agent import NetlistGenerationAgent
from .agents.judge_agent import LLMJudge
from .agents.base_agent import AgentContext

from .models.output import PipelineOutput, StageResult, RawCVOutput, NetlistOutput
from .models.components import ComponentInfo
from .models.wires import WireInfo
from .models.nodes import NodeInfo
from .models.difficulty import DifficultyScores
from .models.training_data import TrainingDataSample

from .config import PipelineConfig


class RefinementPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Optional[PipelineConfig] = None, output_subdir: Optional[str] = None):
        self.config = config or PipelineConfig()
        
        # Output directories
        self.output_dir = Path(self.config.output_dir)
        
        # Use provided subdir or default to agents/
        if output_subdir:
            self.agents_dir = Path(output_subdir)
        else:
            self.agents_dir = self.output_dir / "agents"
        
        self.training_dir = Path(self.config.training_data_dir)
        
        # Create directories
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize agents
        self.agent1 = ComponentRefinementAgent(
            model=self.config.models.agent1_model,
            config=self.config
        )
        
        self.agent2 = ConnectivityRefinementAgent(
            model=self.config.models.agent2_model,
            config=self.config
        )
        
        self.agent3 = NetlistGenerationAgent(
            model=self.config.models.agent3_model,
            config=self.config
        )
        
        self.judge = LLMJudge(
            model=self.config.models.judge_model,
            config=self.config
        )
    
    def run(
        self,
        image_path: str,
        yolo_detections: List[Dict],
        hough_lines: List[Dict],
        junction_candidates: List[Dict],
        original_image: Optional[Image.Image] = None,
        labeled_image: Optional[Image.Image] = None
    ) -> PipelineOutput:
        """Run the complete refinement pipeline"""
        
        start_time = datetime.now()
        
        # Get image dimensions
        if original_image:
            img_width, img_height = original_image.size
        else:
            img = Image.open(image_path)
            img_width, img_height = img.size
        
        # Initialize output
        output = PipelineOutput(image_path=image_path)
        
        # Store raw CV output
        output.raw_cv_output = RawCVOutput(
            yolo_detections=yolo_detections,
            hough_lines=hough_lines,
            junction_candidates=junction_candidates,
            image_width=img_width,
            image_height=img_height,
            image_hash=self._compute_image_hash(original_image) if original_image else ""
        )
        
        print(f"\n{'='*70}")
        print(f"  REFINEMENT PIPELINE v2.0")
        print(f"  Image: {image_path}")
        print(f"  Timestamp: {output.timestamp}")
        print(f"{'='*70}")
        
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 1: Component Refinement
        # ═══════════════════════════════════════════════════════════════════
        
        print(f"\n{'─'*70}")
        print(f"  STAGE 1: Component Refinement ({self.config.models.agent1_model})")
        print(f"{'─'*70}")
        
        agent1_input = {
            "yolo_detections": yolo_detections,
            "image_size": {"width": img_width, "height": img_height}
        }
        
        agent1_output, agent1_result = self.agent1.run(
            input_data=agent1_input
        )
        
        output.set_stage_result("component_refinement", agent1_result)
        
        # Extract components
        components = agent1_output.get("components", [])
        corrected = agent1_output.get("corrected", [])
        agent_added = agent1_output.get("agent_added", [])
        output.components = components
        
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 2: Connectivity Refinement
        # ═══════════════════════════════════════════════════════════════════
        
        print(f"\n{'─'*70}")
        print(f"  STAGE 2: Connectivity Refinement ({self.config.models.agent2_model})")
        if not agent1_result.passed:
            print(f"  ⚠ Upstream needs review (confidence: {agent1_result.final_score:.2f})")
        print(f"{'─'*70}")
        
        agent2_input = {
            "hough_lines": hough_lines,
            "components": components,
            "image_size": {"width": img_width, "height": img_height}
        }
        
        agent2_output, agent2_result = self.agent2.run(
            input_data=agent2_input
        )
        
        output.set_stage_result("connectivity_refinement", agent2_result)
        
        # Extract wires and nodes
        wires = agent2_output.get("wires", [])
        nodes = agent2_output.get("nodes", [])
        output.wires = wires
        output.nodes = nodes
        
        # ═══════════════════════════════════════════════════════════════════
        # STAGE 3: Netlist Generation + Judge Validation Loop
        # ═══════════════════════════════════════════════════════════════════

        print(f"\n{'─'*70}")
        print(f"  STAGE 3: Netlist Generation ({self.config.models.agent3_model})")
        if not agent2_result.passed:
            print(f"  ⚠ Upstream needs review (confidence: {agent2_result.final_score:.2f})")
        print(f"{'─'*70}")

        agent3_input = {
            "components": components,
            "wires": wires,
            "nodes": nodes,
            "image_size": {"width": img_width, "height": img_height}
        }

        # Initial Agent 3 run
        agent3_output, agent3_result = self.agent3.run(
            input_data=agent3_input
        )

        output.set_stage_result("netlist_generation", agent3_result)
        output.netlist = self.agent3.get_netlist_output()

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 4: LLM-as-Judge with Agent 3 Retry Loop
        # ═══════════════════════════════════════════════════════════════════

        print(f"\n{'─'*70}")
        print(f"  STAGE 4: LLM-as-Judge ({self.config.models.judge_model}) [MANDATORY]")
        print(f"{'─'*70}")

        judge_total_iterations = 0
        best_judge_result = None
        best_judge_score = 0.0

        max_judge_retries = self.config.iterations.judge_max_retries

        for judge_attempt in range(1, max_judge_retries + 2):  # +2: initial + max_retries
            judge_total_iterations = judge_attempt

            # Build topology for judge
            netlist_str = output.netlist.spice_code if output.netlist else ""

            if judge_attempt == 1:
                print(f"\n  Initial Judge Evaluation")
            else:
                print(f"\n  Judge Retry {judge_attempt - 1}/{max_judge_retries}")

            judge_result = self.judge.judge(
                topology_components=components,
                topology_nodes=nodes,
                spice_code=netlist_str,
                retry_on_fail=True,
                simulation_logs=sim_logs,
                simulation_success=sim_success,
                image=original_image,
                corrected=corrected,
                agent_added=agent_added,
            )

            print(f"  Verdict: {judge_result.get('verdict')}")
            print(f"  Confidence: {judge_result.get('confidence', 0):.2f}")
            if judge_result.get("issues"):
                print(f"  Issues ({len(judge_result.get('issues', []))}): {judge_result.get('issues')[:3]}")

            # Track best judge result
            current_score = judge_result.get("confidence", 0.0)
            if current_score > best_judge_score:
                best_judge_score = current_score
                best_judge_result = judge_result

            # Check if passed
            if self.judge.is_pass(judge_result):
                print(f"  ✓ PASSED - Judge validation successful")
                best_judge_result = judge_result  # Use the passing result
                break

            # If we've exhausted retries, stop
            if judge_attempt >= max_judge_retries + 1:
                print(f"  ✗ FAILED - Exhausted {max_judge_retries} judge retries")
                break

            # ═══════════════════════════════════════════════════════════
            # Get feedback for Agent 3 retry
            # ═══════════════════════════════════════════════════════════

            print(f"\n  Preparing feedback for Agent 3 retry...")
            feedback = self.judge.get_feedback_for_retry(judge_result)

            fb_agent1 = str(feedback.get("feedback_for_agent1", ""))
            fb_agent2 = str(feedback.get("feedback_for_agent2", ""))

            if fb_agent1:
                print(f"  Feedback for Agent 1: {fb_agent1[:120]}...")
            if fb_agent2:
                print(f"  Feedback for Agent 2: {fb_agent2[:120]}...")

            # ── Re-run Agent 1 only if judge has actionable feedback for it ──
            # Positive-only phrases that indicate no real issue for Agent 1
            _agent1_ok_phrases = [
                "all correct", "all good", "no issues", "correctly identified",
                "looks correct", "no changes needed", "nothing to change",
                "well identified", "accurate", "no corrections needed",
            ]
            fb_agent1_lower = fb_agent1.lower()
            agent1_needs_retry = bool(fb_agent1) and not any(
                phrase in fb_agent1_lower for phrase in _agent1_ok_phrases
            )

            if agent1_needs_retry:
                print(f"\n{'─'*70}")
                print(f"  AGENT 1 RETRY (revision {revision + 1}, with judge feedback)")
                print(f"{'─'*70}")

                # Reset Agent 1 state
                self.agent1.iteration_history = []
                self.agent1.best_output = None
                self.agent1.best_score = 0.0
                self.agent1.best_iteration = 0

                agent1_retry_input = {
                    "yolo_detections": yolo_detections,
                    "image_size": {"width": img_width, "height": img_height},
                    "judge_feedback": fb_agent1,
                }

                agent1_output, agent1_result = self.agent1.run(
                    input_data=agent1_retry_input,
                    images=agent1_images if agent1_images else None,
                )

                output.set_stage_result("component_refinement", agent1_result)

                # Update components from retry
                components = agent1_output.get("components", [])
                corrected = agent1_output.get("corrected", [])
                agent_added = agent1_output.get("agent_added", [])
                output.components = components

                revision_record.agent1_ran = True
                revision_record.agent1_score = agent1_result.final_score
            else:
                print(f"  ↷ Agent 1 skipped — judge has no actionable feedback for component detection")

            # Store feedback for Agent 2 to use in next revision's Agent 2 run
            judge_feedback_for_agent2 = {
                "feedback_for_agent2": fb_agent2,
                "errors": feedback.get("errors", []),
                "confidence": judge_result.get("confidence", 0.0),
                "component_issues": [
                    f"{k}: {v.get('notes', '')}"
                    for k, v in judge_result.get('component_mapping', {}).items()
                    if isinstance(v, dict) and v.get('status') not in ['MATCH', 'OK', 'IMPLICIT']
                ],
                "node_issues": [
                    f"{k}: {v.get('status', 'mismatch')}"
                    for k, v in judge_result.get('node_mapping', {}).items()
                    if isinstance(v, dict) and v.get('status') not in ['MATCH', 'OK']
                ]
            }

            agent3_output, agent3_result = self.agent3.run(
                input_data=agent3_retry_input
            )

            # Update netlist with new output
            output.netlist = self.agent3.get_netlist_output()

            # Merge iteration history (keep both initial + retry iterations)
            output.stage_results["netlist_generation"].iteration_history.extend(
                agent3_result.iteration_history
            )
            output.stage_results["netlist_generation"].iterations = len(
                output.stage_results["netlist_generation"].iteration_history
            )
            output.stage_results["netlist_generation"].final_metrics = agent3_result.final_metrics
            output.stage_results["netlist_generation"].final_score = agent3_result.final_score
            output.stage_results["netlist_generation"].passed = agent3_result.passed

        # ═══════════════════════════════════════════════════════════════════
        # Finalize Judge Results
        # ═══════════════════════════════════════════════════════════════════

        judge_stage = StageResult(
            stage_name="cross_validation",
            model_used=self.config.models.judge_model,
            iterations=judge_total_iterations,
            final_score=best_judge_result.get("confidence", 0.0),
            passed=self.judge.is_pass(best_judge_result)
        )

        if not judge_stage.passed:
            judge_stage.needs_review = True
            judge_stage.review_reasons = best_judge_result.get("issues", [])

        output.set_stage_result("cross_validation", judge_stage)

        print(f"\n{'─'*70}")
        print(f"  JUDGE FINAL RESULT")
        print(f"  Verdict: {best_judge_result.get('verdict')}")
        print(f"  Confidence: {best_judge_result.get('confidence', 0):.2f}")
        print(f"  Total Judge Iterations: {judge_total_iterations}")
        print(f"{'─'*70}")
        
        # ═══════════════════════════════════════════════════════════════════
        # COMPUTE FINAL STATUS
        # ═══════════════════════════════════════════════════════════════════
        
        output.overall_confidence = (
            agent1_result.final_score * 0.25 +
            agent2_result.final_score * 0.25 +
            agent3_result.final_score * 0.25 +
            judge_result.get("confidence", 0.0) * 0.25
        )
        
        output.compute_overall_status()
        
        # Timing
        output.total_time = (datetime.now() - start_time).total_seconds()
        output.total_llm_calls = (
            len(self.agent1.iteration_history) + 
            len(self.agent2.iteration_history) + 
            len(self.agent3.iteration_history) + 1  # +1 for judge
        )
        output.total_tokens = (
            self.agent1.total_tokens +
            self.agent2.total_tokens +
            self.agent3.total_tokens +
            self.judge.total_tokens
        )
        
        # Print summary
        self._print_summary(output)
        
        # Build training sample
        training_sample = self._build_training_sample(output)
        
        # Save outputs
        self._save_outputs(output, training_sample)
        
        return output
    
    def _compute_image_hash(self, image: Image.Image) -> str:
        if image is None:
            return ""
        return hashlib.md5(image.tobytes()).hexdigest()[:16]
    
    def _build_topology_string(
        self, 
        components: List[Any], 
        wires: List[Any], 
        nodes: List[Any]
    ) -> str:
        """Build topology description for judge"""
        lines = ["TOPOLOGY:"]
        lines.append("\nCOMPONENTS:")
        
        for comp in components:
            if hasattr(comp, 'id'):
                cid, ctype = comp.id, comp.type
                terms = [t.name for t in comp.terminals] if comp.terminals else []
            else:
                cid = comp.get('id', '?')
                ctype = comp.get('type', '?')
                terms = [t.get('name', '?') for t in comp.get('terminals', [])]
            
            lines.append(f"  {cid}: {ctype} (terminals: {', '.join(terms)})")
        
        lines.append("\nNODES:")
        for node in nodes:
            nid = node.id if hasattr(node, 'id') else node.get('id', '?')
            ntype = node.type if hasattr(node, 'type') else node.get('type', '?')
            lines.append(f"  {nid}: {ntype}")
        
        return "\n".join(lines)
    
    def _print_summary(self, output: PipelineOutput):
        """Print pipeline summary"""
        print(f"\n{'='*70}")
        print(f"  PIPELINE SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n  Status: {output.overall_status}")
        print(f"  Quality Tier: {output.quality_tier}")
        print(f"  Overall Confidence: {output.overall_confidence:.3f}")
        print(f"  Needs Review: {output.needs_review}")
        
        if output.review_reasons:
            print(f"\n  Review Reasons:")
            for reason in output.review_reasons[:5]:
                print(f"    • {reason[:80]}")
        
        print(f"\n  Stage Results:")
        for name, result in output.stage_results.items():
            status = "✓" if result.passed else "⚠"
            print(f"    {status} {name}: {result.final_score:.3f} ({result.iterations} iterations)")
        
        print(f"\n  Performance:")
        print(f"    Total Time: {output.total_time:.1f}s")
        print(f"    LLM Calls: {output.total_llm_calls}")
        print(f"    Total Tokens: {output.total_tokens}")
        
        print(f"\n{'='*70}")
    
    def _build_training_sample(self, output: PipelineOutput) -> TrainingDataSample:
        """Build training data sample"""
        image_stem = Path(output.image_path).stem
        
        sample = TrainingDataSample(
            sample_id=f"{image_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            image_path=output.image_path
        )
        
        # Set status from output
        sample.pipeline_status = output.overall_status
        sample.quality_tier = output.quality_tier
        sample.overall_confidence = output.overall_confidence
        
        # Add difficulty scores
        sample.difficulty_scores = DifficultyScores()
        
        return sample
    
    def _save_outputs(self, output: PipelineOutput, training_sample: TrainingDataSample):
        """Save all outputs to files"""
        
        image_stem = Path(output.image_path).stem
        
        # ═══════════════════════════════════════════════════════════════════
        # AGENTS JSON
        # ═══════════════════════════════════════════════════════════════════
        
        agents_output = {
            "image_path": output.image_path,
            "timestamp": output.timestamp,
            "overall_status": output.overall_status,
            "overall_confidence": output.overall_confidence,
            "quality_tier": output.quality_tier,
            "needs_review": output.needs_review,
            
            "stage_results": {
                name: {
                    "passed": r.passed,
                    "score": r.final_score,
                    "iterations": r.iterations,
                    "model": r.model_used
                }
                for name, r in output.stage_results.items()
            },
            
            "components": [c.to_dict() if hasattr(c, 'to_dict') else c for c in output.components],
            "wires": [w.to_dict() if hasattr(w, 'to_dict') else w for w in output.wires],
            "nodes": [n.to_dict() if hasattr(n, 'to_dict') else n for n in output.nodes],
            
            "performance": {
                "total_time": output.total_time,
                "llm_calls": output.total_llm_calls,
                "tokens": output.total_tokens
            }
        }
        
        agents_json_path = self.agents_dir / f"{image_stem}_agents.json"
        with open(agents_json_path, 'w') as f:
            json.dump(agents_output, f, indent=2)
        
        # ═══════════════════════════════════════════════════════════════════
        # NETLIST
        # ═══════════════════════════════════════════════════════════════════
        
        if output.netlist and output.netlist.spice_code:
            spice_path = self.agents_dir / f"{image_stem}.sp"
            with open(spice_path, 'w') as f:
                f.write(output.netlist.spice_code)
        
        # ═══════════════════════════════════════════════════════════════════
        # SIMULATION LOGS
        # ═══════════════════════════════════════════════════════════════════
        
        if output.netlist and output.netlist.simulation_logs:
            sim_log_path = self.agents_dir / f"{image_stem}_simulation.log"
            with open(sim_log_path, 'w') as f:
                f.write(f"# Simulation Log for {image_stem}\n")
                f.write(f"# Successful: {output.netlist.simulation_successful}\n\n")
                f.write(output.netlist.simulation_logs)
        
        # ═══════════════════════════════════════════════════════════════════
        # CIRCUIT CAPTION (semantic description)
        # ═══════════════════════════════════════════════════════════════════
        
        caption = self._generate_caption(output)
        caption_path = self.agents_dir / f"{image_stem}_caption.txt"
        with open(caption_path, 'w') as f:
            f.write(caption)
        
        # ═══════════════════════════════════════════════════════════════════
        # LLM FLOW LOG
        # ═══════════════════════════════════════════════════════════════════
        
        flow_log = self._generate_flow_log(output, image_stem)
        flow_path = self.agents_dir / f"{image_stem}_llm_flow.md"
        with open(flow_path, 'w') as f:
            f.write(flow_log)
        
        # ═══════════════════════════════════════════════════════════════════
        # TRAINING DATA
        # ═══════════════════════════════════════════════════════════════════
        
        training_path = self.training_dir / f"{image_stem}_training.json"
        with open(training_path, 'w') as f:
            json.dump(training_sample.to_dict(), f, indent=2)
        
        print(f"\n  📁 Outputs saved to: {self.agents_dir}")
        print(f"     - {image_stem}.sp (netlist)")
        print(f"     - {image_stem}_simulation.log")
        print(f"     - {image_stem}_caption.txt")
        print(f"     - {image_stem}_llm_flow.md")
        print(f"     - {image_stem}_agents.json")
    
    def _generate_caption(self, output: PipelineOutput) -> str:
        """Generate a high-level circuit description"""
        
        # Build component list for LLM
        comp_list = []
        comp_counts = {}
        for comp in output.components:
            ctype = comp.type if hasattr(comp, 'type') else comp.get('type', 'Unknown')
            cid = comp.id if hasattr(comp, 'id') else comp.get('id', '?')
            comp_list.append(f"{cid}: {ctype}")
            comp_counts[ctype] = comp_counts.get(ctype, 0) + 1
        
        # Get netlist
        netlist = output.netlist.spice_code if output.netlist else ""
        
        # Use LLM to generate circuit description
        try:
            from openai import OpenAI
            client = OpenAI()
            
            prompt = f"""Analyze this circuit and provide:
1. Circuit name/type (e.g., "Current Mirror", "Differential Amplifier", "Voltage Divider")
2. Brief description of how it works (2-3 sentences)
3. Key characteristics

Components: {', '.join(comp_list)}

Netlist:
{netlist[:800] if netlist else 'N/A'}

Format your response as:
CIRCUIT: [name]
DESCRIPTION: [how it works]
TOPOLOGY: [input] -> [processing stages] -> [output]
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback
            comp_str = ", ".join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in comp_counts.items())
            return f"Analog circuit with {comp_str}"
    
    def _generate_flow_log(self, output: PipelineOutput, image_stem: str) -> str:
        """Generate LLM flow log in markdown"""
        
        lines = []
        lines.append(f"# Circuit Analysis: {image_stem}")
        lines.append(f"")
        lines.append(f"**Status:** {output.overall_status}")
        lines.append(f"**Confidence:** {output.overall_confidence:.1%}")
        lines.append(f"")
        
        # Components table
        lines.append(f"## Components ({len(output.components)})")
        lines.append(f"")
        lines.append(f"| ID | Type | Source | Confidence |")
        lines.append(f"|---|---|---|---|")
        for comp in output.components:
            cid = comp.id if hasattr(comp, 'id') else comp.get('id', '?')
            ctype = comp.type if hasattr(comp, 'type') else comp.get('type', '?')
            source = comp.source if hasattr(comp, 'source') else comp.get('source', '?')
            conf = comp.confidence if hasattr(comp, 'confidence') else comp.get('confidence', 0)
            lines.append(f"| {cid} | {ctype} | {source} | {conf:.2f} |")
        lines.append(f"")
        
        # Wires summary
        lines.append(f"## Wires ({len(output.wires)})")
        lines.append(f"")
        
        # Netlist
        if output.netlist and output.netlist.spice_code:
            lines.append(f"## Generated Netlist")
            lines.append(f"")
            lines.append(f"```spice")
            lines.append(output.netlist.spice_code)
            lines.append(f"```")
            lines.append(f"")
            sim_status = "✓ PASSED" if output.netlist.simulation_successful else "✗ FAILED"
            lines.append(f"**Simulation:** {sim_status}")
        
        # Stage results
        lines.append(f"")
        lines.append(f"## Pipeline Stages")
        lines.append(f"")
        for name, result in output.stage_results.items():
            status = "✓" if result.passed else "✗"
            lines.append(f"- **{name}**: {status} (score: {result.final_score:.2f})")
        
        return "\n".join(lines)


def run_pipeline(
    image_path: str,
    yolo_detections: List[Dict],
    hough_lines: List[Dict],
    junction_candidates: List[Dict],
    config: Optional[PipelineConfig] = None,
    output_subdir: Optional[str] = None,
    **kwargs
) -> PipelineOutput:
    """Convenience function to run pipeline"""
    
    pipeline = RefinementPipeline(config, output_subdir=output_subdir)
    return pipeline.run(
        image_path=image_path,
        yolo_detections=yolo_detections,
        hough_lines=hough_lines,
        junction_candidates=junction_candidates,
        **kwargs
    )

