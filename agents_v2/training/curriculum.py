"""
Curriculum Learning Utilities

Implements EMA-based curriculum learning for training YOLOv11m
with noisy synthetic data.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class SampleDifficulty:
    """Difficulty tracking for a single sample"""
    sample_id: str
    image_path: str
    
    # Static difficulty (computed once)
    initial_difficulty: float = 0.5
    curriculum_stage: int = 3
    
    # Dynamic difficulty (updated with EMA during training)
    ema_loss: float = 0.0
    ema_loss_component: float = 0.0
    ema_loss_wire: float = 0.0
    ema_loss_node: float = 0.0
    
    # Training history
    times_sampled: int = 0
    last_epoch_sampled: int = -1
    avg_loss_history: List[float] = field(default_factory=list)
    
    # Flags
    needs_review: bool = False
    excluded: bool = False
    
    def update_ema(self, loss: float, component_loss: float = 0.0,
                   wire_loss: float = 0.0, node_loss: float = 0.0,
                   decay: float = 0.9):
        """Update EMA losses after training on this sample"""
        if self.times_sampled == 0:
            # First sample - initialize
            self.ema_loss = loss
            self.ema_loss_component = component_loss
            self.ema_loss_wire = wire_loss
            self.ema_loss_node = node_loss
        else:
            # EMA update
            self.ema_loss = decay * self.ema_loss + (1 - decay) * loss
            self.ema_loss_component = decay * self.ema_loss_component + (1 - decay) * component_loss
            self.ema_loss_wire = decay * self.ema_loss_wire + (1 - decay) * wire_loss
            self.ema_loss_node = decay * self.ema_loss_node + (1 - decay) * node_loss
        
        self.times_sampled += 1
        self.avg_loss_history.append(loss)
    
    @property
    def dynamic_difficulty(self) -> float:
        """Compute dynamic difficulty based on EMA loss"""
        if self.times_sampled == 0:
            return self.initial_difficulty
        
        # Combine static and dynamic difficulty
        # High EMA loss = harder sample
        # Normalize EMA loss to 0-1 range using sigmoid
        normalized_loss = 1.0 / (1.0 + math.exp(-self.ema_loss + 1))
        
        # Blend with initial difficulty
        return 0.3 * self.initial_difficulty + 0.7 * normalized_loss


class DifficultyTracker:
    """
    Tracks sample difficulties and updates EMA during training.
    """
    
    def __init__(self, manifest_path: Optional[str] = None, ema_decay: float = 0.9):
        self.samples: Dict[str, SampleDifficulty] = {}
        self.ema_decay = ema_decay
        self.current_epoch = 0
        
        if manifest_path:
            self.load_manifest(manifest_path)
    
    def load_manifest(self, manifest_path: str):
        """Load difficulty manifest from JSON file"""
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        for sample_data in manifest.get("samples", []):
            sample = SampleDifficulty(
                sample_id=sample_data["sample_id"],
                image_path=sample_data["image_path"],
                initial_difficulty=sample_data.get("overall_difficulty", 0.5),
                curriculum_stage=sample_data.get("curriculum_stage", 3),
                ema_loss_component=sample_data.get("ema_loss_component") or 0.0,
                ema_loss_wire=sample_data.get("ema_loss_wire") or 0.0,
                ema_loss_node=sample_data.get("ema_loss_node") or 0.0,
                needs_review=sample_data.get("needs_review", False)
            )
            self.samples[sample.sample_id] = sample
    
    def save_manifest(self, output_path: str):
        """Save updated difficulty manifest"""
        manifest = {
            "version": "1.0",
            "current_epoch": self.current_epoch,
            "ema_decay": self.ema_decay,
            "samples": []
        }
        
        for sample in self.samples.values():
            manifest["samples"].append({
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "initial_difficulty": sample.initial_difficulty,
                "curriculum_stage": sample.curriculum_stage,
                "dynamic_difficulty": sample.dynamic_difficulty,
                "ema_loss": sample.ema_loss,
                "ema_loss_component": sample.ema_loss_component,
                "ema_loss_wire": sample.ema_loss_wire,
                "ema_loss_node": sample.ema_loss_node,
                "times_sampled": sample.times_sampled,
                "needs_review": sample.needs_review,
                "excluded": sample.excluded
            })
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def update_sample(self, sample_id: str, loss: float,
                      component_loss: float = 0.0,
                      wire_loss: float = 0.0,
                      node_loss: float = 0.0):
        """Update difficulty for a sample after training"""
        if sample_id in self.samples:
            sample = self.samples[sample_id]
            sample.update_ema(loss, component_loss, wire_loss, node_loss, self.ema_decay)
            sample.last_epoch_sampled = self.current_epoch
    
    def get_samples_for_stage(self, stage: int) -> List[str]:
        """Get sample IDs for a curriculum stage"""
        return [
            s.sample_id for s in self.samples.values()
            if s.curriculum_stage == stage and not s.excluded
        ]
    
    def get_samples_up_to_stage(self, max_stage: int) -> List[str]:
        """Get sample IDs for stages 1 through max_stage"""
        return [
            s.sample_id for s in self.samples.values()
            if s.curriculum_stage <= max_stage and not s.excluded
        ]
    
    def advance_epoch(self):
        """Advance to next epoch"""
        self.current_epoch += 1


class CurriculumSampler:
    """
    Curriculum learning sampler for PyTorch DataLoader.
    
    Strategies:
    1. Stage-based: Train on easy samples first, gradually include harder ones
    2. Self-paced: Use EMA loss to dynamically adjust sample weights
    3. Anti-curriculum: Train on hard samples first (for fine-tuning)
    """
    
    def __init__(
        self,
        tracker: DifficultyTracker,
        strategy: str = "stage_based",
        stage_schedule: Optional[Dict[int, int]] = None
    ):
        """
        Args:
            tracker: DifficultyTracker with sample difficulties
            strategy: "stage_based", "self_paced", or "anti_curriculum"
            stage_schedule: Dict mapping stage -> epoch when to include
                           e.g., {1: 0, 2: 5, 3: 10, 4: 15, 5: 20}
        """
        self.tracker = tracker
        self.strategy = strategy
        self.stage_schedule = stage_schedule or {
            1: 0,   # Include stage 1 from epoch 0
            2: 5,   # Include stage 2 from epoch 5
            3: 10,  # Include stage 3 from epoch 10
            4: 15,  # Include stage 4 from epoch 15
            5: 20   # Include stage 5 from epoch 20
        }
    
    def get_sample_ids(self, epoch: int) -> List[str]:
        """Get list of sample IDs to train on for this epoch"""
        
        if self.strategy == "stage_based":
            return self._stage_based_sampling(epoch)
        elif self.strategy == "self_paced":
            return self._self_paced_sampling(epoch)
        elif self.strategy == "anti_curriculum":
            return self._anti_curriculum_sampling(epoch)
        else:
            # All samples
            return list(self.tracker.samples.keys())
    
    def get_sample_weights(self, epoch: int) -> Dict[str, float]:
        """Get sampling weights for each sample"""
        
        weights = {}
        sample_ids = self.get_sample_ids(epoch)
        
        if self.strategy == "self_paced":
            # Weight by inverse of EMA loss (focus on learnable samples)
            for sid in sample_ids:
                sample = self.tracker.samples[sid]
                # Avoid division by zero
                weights[sid] = 1.0 / (sample.ema_loss + 0.1) if sample.times_sampled > 0 else 1.0
        else:
            # Equal weights
            for sid in sample_ids:
                weights[sid] = 1.0
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _stage_based_sampling(self, epoch: int) -> List[str]:
        """Include samples based on curriculum stage schedule"""
        
        # Determine max stage for this epoch
        max_stage = 1
        for stage, start_epoch in sorted(self.stage_schedule.items()):
            if epoch >= start_epoch:
                max_stage = stage
        
        return self.tracker.get_samples_up_to_stage(max_stage)
    
    def _self_paced_sampling(self, epoch: int) -> List[str]:
        """Dynamic sampling based on EMA loss"""
        
        # Start with stage-based, but adjust based on loss
        base_samples = self._stage_based_sampling(epoch)
        
        if epoch < 5:
            # Early epochs: stick to easy samples
            return base_samples
        
        # After warmup: include samples where we're making progress
        # (decreasing loss) even if they're harder
        
        all_samples = list(self.tracker.samples.values())
        selected = set(base_samples)
        
        for sample in all_samples:
            if sample.sample_id in selected:
                continue
            
            # Check if loss is decreasing (learning happening)
            if len(sample.avg_loss_history) >= 2:
                recent_avg = sum(sample.avg_loss_history[-3:]) / min(3, len(sample.avg_loss_history))
                older_avg = sum(sample.avg_loss_history[:-3]) / max(1, len(sample.avg_loss_history) - 3)
                
                if recent_avg < older_avg * 0.9:  # Loss decreased by >10%
                    selected.add(sample.sample_id)
        
        return list(selected)
    
    def _anti_curriculum_sampling(self, epoch: int) -> List[str]:
        """Start with hard samples (for fine-tuning pretrained models)"""
        
        # Reverse the stage schedule
        max_stage = 5
        for stage in sorted(self.stage_schedule.keys(), reverse=True):
            if epoch >= self.stage_schedule[max_stage]:
                break
            max_stage = stage
        
        # Get samples from this stage and above
        return [
            s.sample_id for s in self.tracker.samples.values()
            if s.curriculum_stage >= max_stage and not s.excluded
        ]


def create_pytorch_sampler(
    tracker: DifficultyTracker,
    epoch: int,
    strategy: str = "stage_based",
    **kwargs
):
    """
    Create a PyTorch-compatible sampler for curriculum learning.
    
    Usage:
        tracker = DifficultyTracker("difficulty_manifest.json")
        sampler = create_pytorch_sampler(tracker, epoch=10)
        dataloader = DataLoader(dataset, sampler=sampler, ...)
    """
    try:
        from torch.utils.data import WeightedRandomSampler
    except ImportError:
        raise ImportError("PyTorch is required for this function")
    
    curriculum = CurriculumSampler(tracker, strategy, **kwargs)
    
    sample_ids = curriculum.get_sample_ids(epoch)
    weights = curriculum.get_sample_weights(epoch)
    
    # Map to indices (assuming dataset is ordered by sample_id)
    sample_to_idx = {sid: i for i, sid in enumerate(tracker.samples.keys())}
    
    indices = [sample_to_idx[sid] for sid in sample_ids]
    sample_weights = [weights.get(sid, 0.0) for sid in sample_ids]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(indices),
        replacement=True
    )


# Example training loop integration
"""
# In your training script:

from agents_v2.training import DifficultyTracker, CurriculumSampler

# Initialize
tracker = DifficultyTracker("training_data/difficulty_manifest.json")
curriculum = CurriculumSampler(tracker, strategy="stage_based")

for epoch in range(num_epochs):
    # Get samples for this epoch
    sample_ids = curriculum.get_sample_ids(epoch)
    
    # Create subset dataset
    epoch_dataset = Subset(full_dataset, 
                          [dataset.sample_to_idx[sid] for sid in sample_ids])
    
    # Train
    for batch in DataLoader(epoch_dataset, ...):
        loss, component_loss, wire_loss, node_loss = train_step(batch)
        
        # Update difficulty tracking
        for sample_id in batch.sample_ids:
            tracker.update_sample(
                sample_id, 
                loss=loss.item(),
                component_loss=component_loss.item(),
                wire_loss=wire_loss.item(),
                node_loss=node_loss.item()
            )
    
    tracker.advance_epoch()
    
    # Save updated manifest periodically
    if epoch % 5 == 0:
        tracker.save_manifest("training_data/difficulty_manifest.json")
"""

