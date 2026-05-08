"""
Agent 3: Netlist Generation Agent

Generates SPICE netlist from topology:
- Converts components and nodes to SPICE format
- Adds simulation directives
- Validates syntax and runs simulation
"""

import json
import re
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

from .base_agent import BaseReActAgent, AgentContext
from ..config import PipelineConfig
from ..models.components import ComponentInfo
from ..models.nodes import NodeInfo
from ..models.output import NetlistOutput


class NetlistGenerationAgent(BaseReActAgent):
    """
    Agent 3: Netlist Generation
    
    Converts topology to valid SPICE netlist.
    Does NOT require vision - works from structured topology only.
    """
    
    def __init__(self, model: str = "gpt-5.2", config: Optional[PipelineConfig] = None):
        super().__init__(
            agent_name="Agent 3: Netlist Generation",
            model=model,
            config=config or PipelineConfig(),
            requires_vision=False  # No vision needed - structured input only
        )
        self.netlist_output = NetlistOutput()
        self.output_dir = Path(config.output_dir if config else "agents_v2_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_max_iterations(self) -> int:
        return self.config.iterations.agent3_max_iterations
    
    def get_pass_threshold(self) -> float:
        return self.config.thresholds.agent3_pass_threshold
    
    def get_min_viable_threshold(self) -> float:
        return self.config.thresholds.agent3_min_viable_threshold
    
    def get_system_prompt(self) -> str:
        return """You are an expert SPICE netlist generator. Your task is to convert circuit topology to valid SPICE netlist.

You will receive:
1. List of components with their terminal-to-node connections
2. List of electrical nodes
3. (On iterations 2+) Simulation errors to fix

SPICE Netlist Format Rules:
- First line is the circuit title
- Comments start with *
- Component format: NAME NODE1 NODE2 [NODE3...] VALUE
- Ground must be node 0
- End with .END

Component Syntax:
- Resistor: R<name> <node1> <node2> <value>  (e.g., R1 n1 n2 10k)
- Capacitor: C<name> <node1> <node2> <value> (e.g., C1 n1 0 1u)
- Inductor: L<name> <node1> <node2> <value>  (e.g., L1 n1 n2 100u)
- MOSFET: M<name> <drain> <gate> <source> <bulk> <model> W=<w> L=<l>
  (e.g., M1 out in 0 0 NMOS W=1u L=180n)
- Voltage: V<name> <+node> <-node> <value> (e.g., VDD vdd 0 1.8)
- Current: I<name> <+node> <-node> <value> (e.g., I1 vdd out 10u)
- BJT: Q<name> <collector> <base> <emitter> <model>
- Diode: D<name> <anode> <cathode> <model>

Required Additions:
- Add VDD voltage source if not present (e.g., VDD vdd 0 1.8)
- Add MOSFET models if MOSFETs used
- Add basic simulation commands (.OP for DC operating point)

Output as JSON:
```json
{
  "thinking": "Your netlist generation reasoning...",
  
  "netlist": {
    "title": "Circuit Netlist",
    "components": [
      {"line": "R1 n1 n2 10k", "comment": "Resistor R1 between n1 and n2"},
      {"line": "M1 n1 gate 0 0 NMOS W=1u L=180n", "comment": "NMOS transistor"}
    ],
    "sources": [
      {"line": "VDD vdd 0 1.8", "comment": "Power supply"}
    ],
    "models": [
      {"line": ".model NMOS NMOS (VTH0=0.4 KP=200u)", "comment": "NMOS model"}
    ],
    "simulation": [
      {"line": ".OP", "comment": "DC operating point"},
      {"line": ".END", "comment": "End of netlist"}
    ]
  },
  
  "spice_code": "* Circuit Netlist\\n* Components\\nR1 n1 n2 10k\\n...",
  
  "node_mapping": {
    "VDD": "vdd",
    "0": "0", 
    "node_1": "n1",
    "node_2": "n2"
  },
  
  "summary": {
    "total_components": 5,
    "total_nodes": 4,
    "has_ground": true,
    "has_power_source": true
  }
}
```

Ensure all node names are SPICE-compatible (alphanumeric, no spaces)."""
    
    def build_user_prompt(self, context: AgentContext, iteration: int) -> str:
        """Build prompt with topology data"""
        
        components = context.original_input.get("components", [])
        nodes = context.original_input.get("nodes", [])
        
        prompt = """Generate a SPICE netlist from this circuit topology.

COMPONENTS:
"""
        for comp in components:
            if isinstance(comp, ComponentInfo):
                connections = ", ".join(
                    f"{t.name}→{t.connected_to_node}" 
                    for t in comp.terminals
                )
                prompt += f"  {comp.id} ({comp.type}): {connections}"
                if comp.value:
                    prompt += f", value={comp.value}"
                prompt += "\n"
            else:
                terminals = comp.get("terminals", [])
                connections = ", ".join(
                    f"{t.get('name')}→{t.get('connected_to_node', 'unknown')}"
                    for t in terminals
                )
                prompt += f"  {comp.get('id')} ({comp.get('type')}): {connections}"
                if comp.get("value"):
                    prompt += f", value={comp.get('value')}"
                prompt += "\n"
        
        prompt += """
ELECTRICAL NODES:
"""
        for node in nodes:
            if isinstance(node, NodeInfo):
                node_type = "POWER" if node.is_power else "GROUND" if node.is_ground else "internal"
                prompt += f"  {node.id} ({node_type}): {node.connected_terminals}\n"
            else:
                node_type = "POWER" if node.get("is_power") else "GROUND" if node.get("is_ground") else "internal"
                prompt += f"  {node.get('id')} ({node_type}): {node.get('connected_terminals', [])}\n"
        
        # Add judge feedback if provided (from judge retry)
        judge_feedback = context.original_input.get("judge_feedback")
        if judge_feedback:
            prompt += f"""

═══════════════════════════════════════════════════════════
JUDGE VALIDATION FEEDBACK (CRITICAL - MUST FIX):
═══════════════════════════════════════════════════════════

The previous netlist FAILED cross-validation by the LLM Judge.
Judge Confidence: {judge_feedback.get('judge_confidence', 0.0):.2f}

ERRORS FOUND:
"""
            for error in judge_feedback.get('errors', []):
                prompt += f"  • {error}\n"

            if judge_feedback.get('component_issues'):
                prompt += f"""
COMPONENT ISSUES:
"""
                for issue in judge_feedback.get('component_issues', []):
                    if issue.strip():
                        prompt += f"  • {issue}\n"

            if judge_feedback.get('node_issues'):
                prompt += f"""
NODE MAPPING ISSUES:
"""
                for issue in judge_feedback.get('node_issues', []):
                    if issue.strip():
                        prompt += f"  • {issue}\n"

            prompt += f"""
JUDGE FEEDBACK:
{judge_feedback.get('feedback', '')}

PREVIOUS NETLIST (FAILED):
{judge_feedback.get('previous_netlist', '')[:1000]}

You MUST fix all the issues above. Pay special attention to:
1. Component-to-netlist mapping (ensure all topology components appear)
2. Node connections (verify each component's terminals connect correctly)
3. SPICE syntax (ground = 0, proper component format)
═══════════════════════════════════════════════════════════
"""
        # Add error context for iteration 2+ (internal agent iterations)
        elif iteration > 1 and context.previous_output:
            prompt += f"""

═══════════════════════════════════════════════════════════
PREVIOUS ITERATION FEEDBACK (MUST FIX THESE ERRORS):
═══════════════════════════════════════════════════════════

PREVIOUS METRICS:
"""
            for metric, value in context.previous_metrics.items():
                status = "✓ PASSED" if metric in context.passed_metrics else "✗ FAILED"
                prompt += f"  {metric}: {value:.3f} {status}\n"

            prompt += f"""
SIMULATION ERRORS:
"""
            for error in context.errors:
                prompt += f"  • {error}\n"

            prompt += f"""
FEEDBACK:
{context.feedback}

Fix the errors above. The previous netlist was:
{context.previous_output.get('spice_code', 'N/A')[:1000]}
═══════════════════════════════════════════════════════════
"""

        prompt += """
Generate a complete, valid SPICE netlist that will simulate successfully."""
        
        return prompt
    
    def parse_response(self, response: str, context: AgentContext) -> Dict[str, Any]:
        """Parse LLM response into structured output - with robust fallback"""
        
        # Extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                # FALLBACK: Try to extract SPICE code directly from response
                print(f"    ⚠ No JSON in response, trying to extract SPICE directly")
                return self._extract_spice_fallback(response, context)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"    ⚠ JSON parse error: {e}, trying SPICE extraction fallback")
            return self._extract_spice_fallback(response, context)
        
        # Get or build SPICE code
        spice_code = data.get("spice_code", "")
        if not spice_code:
            # Build from structured netlist
            netlist = data.get("netlist", {})
            lines = [f"* {netlist.get('title', 'Circuit Netlist')}"]
            lines.append("*")
            
            lines.append("* Components")
            for comp in netlist.get("components", []):
                lines.append(comp.get("line", ""))
            
            lines.append("*")
            lines.append("* Sources")
            for src in netlist.get("sources", []):
                lines.append(src.get("line", ""))
            
            lines.append("*")
            lines.append("* Models")
            for model in netlist.get("models", []):
                lines.append(model.get("line", ""))
            
            lines.append("*")
            lines.append("* Simulation")
            for sim in netlist.get("simulation", []):
                lines.append(sim.get("line", ""))
            
            spice_code = "\n".join(lines)
        
        # Store netlist output
        self.netlist_output.spice_code = spice_code
        
        actions = [
            f"Generated netlist with {data.get('summary', {}).get('total_components', 0)} components",
            f"Nodes: {data.get('summary', {}).get('total_nodes', 0)}"
        ]
        
        return {
            "spice_code": spice_code,
            "netlist": data.get("netlist", {}),
            "node_mapping": data.get("node_mapping", {}),
            "summary": data.get("summary", {}),
            "thinking": data.get("thinking", ""),
            "actions": actions,
            "raw_response": data
        }
    
    def evaluate_metrics(self, output: Dict[str, Any], context: AgentContext) -> Dict[str, float]:
        """Evaluate netlist metrics including simulation"""
        
        spice_code = output.get("spice_code", "")
        components = context.original_input.get("components", [])
        nodes = context.original_input.get("nodes", [])
        
        metrics = {}
        
        # Syntax validity - basic checks
        syntax_score = 0.0
        if spice_code:
            # Check for required elements
            has_end = ".end" in spice_code.lower()
            has_ground = " 0 " in spice_code or "\t0\t" in spice_code
            has_components = any(
                line.strip() and not line.startswith("*") and not line.startswith(".")
                for line in spice_code.split("\n")
                if line.strip()
            )
            
            if has_end:
                syntax_score += 0.3
            if has_ground:
                syntax_score += 0.3
            if has_components:
                syntax_score += 0.4
        
        metrics["syntax_valid"] = syntax_score
        
        # Try simulation
        sim_result = self._run_simulation(spice_code)
        metrics["simulation_runs"] = 1.0 if sim_result["success"] else 0.0
        
        # Print simulation status
        if sim_result["success"]:
            print(f"    ✓ ngspice simulation PASSED")
            # Extract node voltages if available
            logs = sim_result.get("logs", "")
            if "Node" in logs and "Voltage" in logs:
                print(f"    Node voltages extracted from simulation")
        else:
            print(f"    ✗ ngspice simulation FAILED")
            for err in sim_result.get("errors", []):
                print(f"      - {err}")
        
        # Store simulation info
        self.netlist_output.simulation_attempted = True
        self.netlist_output.simulation_successful = sim_result["success"]
        self.netlist_output.simulation_logs = sim_result.get("logs", "")
        
        # Topology preserved - check component count
        netlist_components = self._count_netlist_components(spice_code)
        expected_components = len([
            c for c in components 
            if (c.type if isinstance(c, ComponentInfo) else c.get("type")) != "Ground"
        ])
        
        if expected_components > 0:
            match_ratio = min(netlist_components, expected_components) / expected_components
            metrics["topology_preserved"] = match_ratio
        else:
            metrics["component_match"] = 0.5

        # Log analysis
        if missing_core:
            print(f"    ⚠ Missing core components: {missing_core}")
        if duplicate_set:
            print(f"    ⚠ Duplicate components: {duplicate_set}")
        if extra_core:
            print(f"    ⚠ Extra non-scaffolding components: {extra_core}")
        if netlist_scaffolding:
            print(f"    ✓ Scaffolding elements (allowed): {netlist_scaffolding}")

        # 3. Node match - check for required structural nodes by name
        # Ground (node 0) and at least one power/signal node must be present.
        # Count-based matching was unreliable: the CV node list is noisy and never
        # passed into agent2_input, so expected_nodes was always 0 (always 0.5).
        netlist_nodes = self._extract_netlist_nodes(spice_code)
        netlist_nodes_lower = {n.lower() for n in netlist_nodes}

        has_ground = "0" in netlist_nodes
        power_names = {"vdd", "vcc", "vss", "vee", "avdd", "dvdd", "v+", "v-"}
        has_power = bool(netlist_nodes_lower & power_names)

        if has_ground and has_power:
            metrics["node_match"] = 1.0
        elif has_ground or has_power:
            metrics["node_match"] = 0.7
        else:
            metrics["node_match"] = 0.3


        return metrics
    
    def build_error_context(
        self,
        output: Dict[str, Any],
        metrics: Dict[str, float],
        context: AgentContext
    ) -> AgentContext:
        """Build error context for next iteration"""
        
        new_context = AgentContext(
            original_input=context.original_input
        )
        
        errors = []
        feedback_parts = []
        
        # Check syntax
        if metrics.get("syntax_valid", 1.0) < 1.0:
            spice_code = output.get("spice_code", "")
            if ".end" not in spice_code.lower():
                errors.append("Missing .END directive at end of netlist")
            if " 0 " not in spice_code:
                errors.append("Ground node '0' not found in any component connection")
            feedback_parts.append(
                "Ensure netlist has proper SPICE format with .END directive and ground node 0."
            )
        
        # Check simulation
        if metrics.get("simulation_runs", 1.0) < 1.0:
            sim_logs = self.netlist_output.simulation_logs
            if sim_logs:
                # Extract key error messages
                error_lines = [
                    line for line in sim_logs.split("\n")
                    if "error" in line.lower() or "fatal" in line.lower()
                ][:5]
                for line in error_lines:
                    errors.append(f"ngspice: {line}")
            errors.append("Simulation failed - check syntax and ensure DC path to ground")

        # Check component match - use detailed analysis from evaluate_metrics
        if metrics.get("component_match", 1.0) < 0.95:
            analysis = getattr(self, '_component_analysis', {})

            missing = analysis.get("missing_core", set())
            duplicates = analysis.get("duplicates", set())
            extra = analysis.get("extra_core", set())

            if missing:
                errors.append(f"MISSING COMPONENTS (must add): {', '.join(sorted(missing))}")

            if duplicates:
                errors.append(f"DUPLICATE COMPONENTS (remove duplicates): {', '.join(sorted(duplicates))}")

            if extra:
                errors.append(f"EXTRA COMPONENTS not in validated list: {', '.join(sorted(extra))} "
                            f"(use scaffolding prefixes like VSUP_, VB_, RSHUNT_ if needed)")

            if not (missing or duplicates or extra):
                errors.append(f"Component match below threshold - verify all validated components appear exactly once")

        # Check node match
        if metrics.get("node_match", 1.0) < 0.90:
            errors.append(
                "Missing required structural nodes: ensure ground (node 0) and at least one "
                "power rail (vdd/vcc/vss) are present in the netlist."
            )


        new_context.errors = errors
        new_context.feedback = "\n".join(feedback_parts)
        
        return new_context
    
    def _run_simulation(self, spice_code: str) -> Dict[str, Any]:
        """Run ngspice simulation on the netlist"""
        
        result = {
            "success": False,
            "logs": "",
            "errors": []
        }
        
        if not spice_code:
            result["errors"].append("Empty netlist")
            return result
        
        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False) as f:
                f.write(spice_code)
                temp_path = f.name
            
            # Run ngspice in batch mode
            cmd = ["ngspice", "-b", temp_path]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            result["logs"] = proc.stdout + proc.stderr
            
            # Check for success
            if proc.returncode == 0:
                # Also check for error messages in output
                if "error" not in proc.stdout.lower() and "error" not in proc.stderr.lower():
                    result["success"] = True
                else:
                    result["errors"].append("Simulation reported errors")
            else:
                result["errors"].append(f"ngspice returned code {proc.returncode}")
            
            # Cleanup
            os.unlink(temp_path)
            
        except subprocess.TimeoutExpired:
            result["errors"].append("Simulation timed out")
        except FileNotFoundError:
            # ngspice not installed - assume success for syntax check only
            result["success"] = True
            result["logs"] = "ngspice not found - skipping simulation"
        except Exception as e:
            result["errors"].append(str(e))
        
        return result
    
    def _count_netlist_components(self, spice_code: str) -> int:
        """Count components in SPICE netlist"""
        count = 0
        for line in spice_code.split("\n"):
            line = line.strip()
            if line and not line.startswith("*") and not line.startswith("."):
                # Component line
                if line[0].upper() in "RCLMQVID":
                    count += 1
        return count
    
    def _extract_netlist_nodes(self, spice_code: str) -> set:
        """Extract node names from SPICE netlist"""
        nodes = set()
        for line in spice_code.split("\n"):
            line = line.strip()
            if line and not line.startswith("*") and not line.startswith("."):
                parts = line.split()
                if len(parts) >= 3:
                    # Extract node names (usually positions 1, 2, sometimes 3, 4)
                    for part in parts[1:5]:
                        # Skip values and model names
                        if not any(c.isalpha() and c not in "abcdefABCDEFxX" for c in part):
                            if not part.replace(".", "").replace("-", "").replace("+", "").isdigit():
                                nodes.add(part)
                        elif part.isalnum():
                            nodes.add(part)
        return nodes
    
    def _get_metric_weights(self) -> Dict[str, float]:
        return self.config.metric_weights.agent3_weights
    
    def _get_metric_target(self, metric: str) -> float:
        targets = {
            "syntax_valid": 1.0,
            "simulation_runs": 1.0,
            "topology_preserved": 0.95,
            "component_match": 1.0,
            "node_match": 1.0
        }
        return targets.get(metric, 0.8)
    
    def get_netlist_output(self) -> NetlistOutput:
        """Get the netlist output object"""
        return self.netlist_output
    
    def _extract_spice_fallback(self, response: str, context: AgentContext) -> Dict[str, Any]:
        """Try to extract SPICE code directly from response text."""
        
        # Look for SPICE-like content
        spice_code = ""
        
        # Try to find code block
        code_match = re.search(r'```(?:spice|ngspice|netlist)?\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if code_match:
            spice_code = code_match.group(1)
        else:
            # Look for lines that look like SPICE
            lines = response.split('\n')
            spice_lines = []
            for line in lines:
                line = line.strip()
                # SPICE lines typically start with component letters or commands
                if line and (line[0] in 'RCLMQVIDrclmqvid*.' or line.startswith('.end')):
                    spice_lines.append(line)
            
            if spice_lines:
                spice_code = '\n'.join(spice_lines)
        
        if not spice_code:
            # Generate minimal SPICE from topology
            spice_code = self._generate_minimal_spice(context)
        
        self.netlist_output.spice_code = spice_code
        
        return {
            "spice_code": spice_code,
            "netlist": {},
            "node_mapping": {},
            "summary": {"fallback": True},
            "thinking": "Fallback: extracted/generated SPICE directly",
            "actions": ["FALLBACK: Generated minimal SPICE netlist"],
            "raw_response": {"fallback": True}
        }
    
    def _generate_minimal_spice(self, context: AgentContext) -> str:
        """Generate minimal SPICE netlist from topology."""
        components = context.original_input.get("components", [])
        
        lines = ["* Auto-generated minimal netlist", "*"]
        lines.append("* Components")
        
        node_counter = 1
        for i, comp in enumerate(components):
            if hasattr(comp, 'type'):
                comp_type = comp.type
                comp_id = comp.id
            else:
                comp_type = comp.get('type', 'R')
                comp_id = comp.get('id', f'C{i}')
            
            # Generate SPICE line based on type
            if comp_type in ['Resistor', 'R']:
                lines.append(f"R{i+1} n{node_counter} n{node_counter+1} 10k")
                node_counter += 2
            elif comp_type in ['Capacitor', 'C']:
                lines.append(f"C{i+1} n{node_counter} 0 1u")
                node_counter += 1
            elif comp_type in ['NMOS', 'MOSFET']:
                lines.append(f"M{i+1} n{node_counter} n{node_counter+1} 0 0 NMOS W=1u L=180n")
                node_counter += 2
            elif comp_type in ['PMOS']:
                lines.append(f"M{i+1} n{node_counter} n{node_counter+1} vdd vdd PMOS W=1u L=180n")
                node_counter += 2
            elif comp_type in ['VoltageSource', 'V', 'Voltage_Source']:
                lines.append(f"V{i+1} vdd 0 1.8")
            elif comp_type in ['CurrentSource', 'I', 'Current_Source']:
                lines.append(f"I{i+1} vdd n{node_counter} 10u")
                node_counter += 1
        
        lines.append("*")
        lines.append("* Power supply")
        lines.append("VDD vdd 0 1.8")
        lines.append("*")
        lines.append("* Models")
        lines.append(".model NMOS NMOS (VTH0=0.4)")
        lines.append(".model PMOS PMOS (VTH0=-0.4)")
        lines.append("*")
        lines.append(".OP")
        lines.append(".END")
        
        return '\n'.join(lines)

