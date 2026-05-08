"""
Training Data Exporter

Exports agent-refined data to formats suitable for training:
- YOLO format for object detection
- COCO format for multi-task learning
- Custom format with difficulty scores for curriculum learning
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..models.training_data import TrainingDataSample, ComponentAnnotation
from ..config import ExportConfig


@dataclass
class ExportStats:
    """Statistics from export operation"""
    total_samples: int = 0
    exported_samples: int = 0
    skipped_samples: int = 0
    total_components: int = 0
    total_wires: int = 0
    total_nodes: int = 0
    samples_by_stage: Dict[int, int] = None
    samples_needing_review: int = 0
    
    def __post_init__(self):
        if self.samples_by_stage is None:
            self.samples_by_stage = {}


class TrainingDataExporter:
    """
    Exports training data in multiple formats for multi-head YOLO training.
    """
    
    def __init__(self, export_config: Optional[ExportConfig] = None):
        self.config = export_config or ExportConfig()
    
    def export_yolo(
        self,
        samples: List[TrainingDataSample],
        output_dir: str,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        min_quality: str = "LOW_CONFIDENCE",
        copy_images: bool = True
    ) -> ExportStats:
        """
        Export samples to YOLO format.
        
        Args:
            samples: List of training samples
            output_dir: Output directory for YOLO dataset
            split_ratio: (train, val, test) split ratios
            min_quality: Minimum quality tier to include
            copy_images: Whether to copy images to output directory
        
        Returns:
            ExportStats with export statistics
        """
        
        output_path = Path(output_dir)
        stats = ExportStats()
        
        # Quality tier ordering
        quality_order = ["FAILED", "LOW_CONFIDENCE", "CONFIDENT", "VERIFIED"]
        min_quality_idx = quality_order.index(min_quality)
        
        # Filter samples by quality
        valid_samples = []
        for sample in samples:
            sample_quality_idx = quality_order.index(sample.quality_tier) if sample.quality_tier in quality_order else 0
            if sample_quality_idx >= min_quality_idx:
                valid_samples.append(sample)
                stats.total_components += len(sample.component_annotations)
                stats.total_wires += len(sample.wire_annotations)
                stats.total_nodes += len(sample.node_annotations)
                
                # Track by curriculum stage
                if sample.difficulty_scores:
                    stage = sample.difficulty_scores.curriculum_stage
                    stats.samples_by_stage[stage] = stats.samples_by_stage.get(stage, 0) + 1
                
                if sample.needs_review:
                    stats.samples_needing_review += 1
            else:
                stats.skipped_samples += 1
        
        stats.total_samples = len(samples)
        stats.exported_samples = len(valid_samples)
        
        # Split samples
        import random
        random.shuffle(valid_samples)
        
        n = len(valid_samples)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])
        
        splits = {
            "train": valid_samples[:train_end],
            "val": valid_samples[train_end:val_end],
            "test": valid_samples[val_end:]
        }
        
        # Create directory structure
        for split in ["train", "val", "test"]:
            (output_path / split / "images").mkdir(parents=True, exist_ok=True)
            (output_path / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Export samples
        for split, split_samples in splits.items():
            for sample in split_samples:
                self._export_yolo_sample(
                    sample, 
                    output_path / split,
                    copy_images
                )
        
        # Create dataset.yaml
        self._create_yolo_yaml(output_path)
        
        # Create difficulty manifest for curriculum learning
        self._create_difficulty_manifest(valid_samples, output_path)
        
        return stats
    
    def _export_yolo_sample(
        self,
        sample: TrainingDataSample,
        split_path: Path,
        copy_images: bool
    ):
        """Export a single sample to YOLO format"""
        
        image_name = Path(sample.image_path).stem
        
        # Copy/link image
        if copy_images:
            src_image = Path(sample.image_path)
            if src_image.exists():
                dst_image = split_path / "images" / f"{image_name}.jpg"
                shutil.copy2(src_image, dst_image)
        
        # Create label file
        label_path = split_path / "labels" / f"{image_name}.txt"
        
        lines = []
        for ann in sample.component_annotations:
            lines.append(ann.to_yolo_line())
        
        with open(label_path, 'w') as f:
            f.write("\n".join(lines))
    
    def _create_yolo_yaml(self, output_path: Path):
        """Create YOLO dataset configuration file"""
        
        yaml_content = f"""# YOLOv11m Multi-Head Training Dataset
# Generated by Agents V2 Pipeline

path: {output_path.absolute()}
train: train/images
val: val/images
test: test/images

# Component Detection Head Classes
names:
"""
        for name, idx in sorted(self.config.component_classes.items(), key=lambda x: x[1]):
            yaml_content += f"  {idx}: {name}\n"
        
        yaml_content += """
# Additional heads (wire, node detection) use separate configs
"""
        
        with open(output_path / "dataset.yaml", 'w') as f:
            f.write(yaml_content)
    
    def _create_difficulty_manifest(
        self,
        samples: List[TrainingDataSample],
        output_path: Path
    ):
        """Create difficulty manifest for curriculum learning"""
        
        manifest = {
            "version": "1.0",
            "description": "Difficulty scores for curriculum learning with EMA",
            "stages": {
                1: {"name": "easiest", "threshold": 0.2, "samples": []},
                2: {"name": "easy", "threshold": 0.4, "samples": []},
                3: {"name": "medium", "threshold": 0.6, "samples": []},
                4: {"name": "hard", "threshold": 0.8, "samples": []},
                5: {"name": "hardest", "threshold": 1.0, "samples": []}
            },
            "samples": []
        }
        
        for sample in samples:
            sample_entry = {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "needs_review": sample.needs_review
            }
            
            if sample.difficulty_scores:
                sample_entry.update({
                    "overall_difficulty": sample.difficulty_scores.overall_difficulty,
                    "component_difficulty": sample.difficulty_scores.component_difficulty,
                    "wire_difficulty": sample.difficulty_scores.wire_difficulty,
                    "node_difficulty": sample.difficulty_scores.node_difficulty,
                    "curriculum_stage": sample.difficulty_scores.curriculum_stage,
                    "ema_loss_component": sample.difficulty_scores.ema_loss_component,
                    "ema_loss_wire": sample.difficulty_scores.ema_loss_wire,
                    "ema_loss_node": sample.difficulty_scores.ema_loss_node
                })
                
                stage = sample.difficulty_scores.curriculum_stage
                manifest["stages"][stage]["samples"].append(sample.sample_id)
            else:
                sample_entry["curriculum_stage"] = 3  # Default to medium
                manifest["stages"][3]["samples"].append(sample.sample_id)
            
            manifest["samples"].append(sample_entry)
        
        with open(output_path / "difficulty_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def export_coco(
        self,
        samples: List[TrainingDataSample],
        output_dir: str,
        min_quality: str = "LOW_CONFIDENCE"
    ) -> ExportStats:
        """
        Export samples to COCO format for multi-task training.
        
        Args:
            samples: List of training samples
            output_dir: Output directory
            min_quality: Minimum quality tier
        
        Returns:
            ExportStats
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stats = ExportStats()
        
        # Build COCO structure
        coco_data = {
            "info": {
                "description": "Circuit Schematic Dataset - Agent Refined",
                "version": "2.0",
                "year": 2024,
                "contributor": "Agents V2 Pipeline"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        for name, idx in self.config.component_classes.items():
            coco_data["categories"].append({
                "id": idx,
                "name": name,
                "supercategory": "component"
            })
        
        # Quality filter
        quality_order = ["FAILED", "LOW_CONFIDENCE", "CONFIDENT", "VERIFIED"]
        min_quality_idx = quality_order.index(min_quality)
        
        annotation_id = 0
        
        for image_id, sample in enumerate(samples):
            # Quality check
            sample_quality_idx = quality_order.index(sample.quality_tier) if sample.quality_tier in quality_order else 0
            if sample_quality_idx < min_quality_idx:
                stats.skipped_samples += 1
                continue
            
            stats.exported_samples += 1
            
            # Add image entry
            coco_data["images"].append({
                "id": image_id,
                "file_name": sample.image_path,
                "width": sample.image_width,
                "height": sample.image_height,
                "difficulty": sample.difficulty_scores.overall_difficulty if sample.difficulty_scores else 0.5,
                "curriculum_stage": sample.difficulty_scores.curriculum_stage if sample.difficulty_scores else 3,
                "needs_review": sample.needs_review
            })
            
            # Add annotations
            for ann in sample.component_annotations:
                x = (ann.x_center - ann.width / 2) * sample.image_width
                y = (ann.y_center - ann.height / 2) * sample.image_height
                w = ann.width * sample.image_width
                h = ann.height * sample.image_height
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": ann.class_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "attributes": {
                        "orientation": ann.orientation,
                        "source": ann.source,
                        "confidence": ann.confidence,
                        "detection_difficulty": ann.detection_difficulty
                    }
                })
                annotation_id += 1
                stats.total_components += 1
        
        stats.total_samples = len(samples)
        
        # Save COCO JSON
        with open(output_path / "annotations.json", 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        return stats
    
    def export_multihead(
        self,
        samples: List[TrainingDataSample],
        output_dir: str
    ) -> ExportStats:
        """
        Export data for multi-head YOLOv11m training.
        
        Creates separate annotation files for each head:
        - components.txt: Component bounding boxes
        - wires.txt: Wire segment endpoints
        - nodes.txt: Node/junction points
        - terminals.txt: Terminal keypoints
        
        Args:
            samples: Training samples
            output_dir: Output directory
        
        Returns:
            ExportStats
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stats = ExportStats()
        
        for sample in samples:
            if sample.quality_tier == "FAILED":
                stats.skipped_samples += 1
                continue
            
            sample_name = Path(sample.image_path).stem
            
            # Component annotations (standard YOLO format)
            comp_path = output_path / "components" / f"{sample_name}.txt"
            comp_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(comp_path, 'w') as f:
                for ann in sample.component_annotations:
                    f.write(ann.to_yolo_line() + "\n")
            stats.total_components += len(sample.component_annotations)
            
            # Wire annotations (line segment format)
            wire_path = output_path / "wires" / f"{sample_name}.txt"
            wire_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(wire_path, 'w') as f:
                for wire_ann in sample.wire_annotations:
                    for seg in wire_ann.segments:
                        # Format: class_id x1 y1 x2 y2
                        x1, y1, x2, y2 = seg.to_normalized(
                            sample.image_width, sample.image_height
                        )
                        class_id = self.config.wire_classes.get(seg.segment_type, 0)
                        f.write(f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n")
            stats.total_wires += len(sample.wire_annotations)
            
            # Node annotations (keypoint format)
            node_path = output_path / "nodes" / f"{sample_name}.txt"
            node_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(node_path, 'w') as f:
                for node_ann in sample.node_annotations:
                    # Format: class_id x y visibility
                    class_id = self.config.node_classes.get(node_ann.node_type, 0)
                    f.write(f"{class_id} {node_ann.x:.6f} {node_ann.y:.6f} 2\n")  # 2 = visible
            stats.total_nodes += len(sample.node_annotations)
            
            # Terminal annotations (keypoint format with parent component)
            term_path = output_path / "terminals" / f"{sample_name}.txt"
            term_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(term_path, 'w') as f:
                for term_ann in sample.terminal_annotations:
                    # Format: class_id x y visibility component_id terminal_name
                    f.write(f"{term_ann.class_id} {term_ann.x:.6f} {term_ann.y:.6f} 2 "
                           f"{term_ann.component_id} {term_ann.terminal_name}\n")
            
            stats.exported_samples += 1
            
            if sample.needs_review:
                stats.samples_needing_review += 1
            
            if sample.difficulty_scores:
                stage = sample.difficulty_scores.curriculum_stage
                stats.samples_by_stage[stage] = stats.samples_by_stage.get(stage, 0) + 1
        
        stats.total_samples = len(samples)
        
        # Create config file for multi-head training
        self._create_multihead_config(output_path)
        
        return stats
    
    def _create_multihead_config(self, output_path: Path):
        """Create configuration for multi-head YOLOv11m training"""
        
        config = {
            "model": "yolov11m",
            "heads": {
                "component_detection": {
                    "type": "detection",
                    "annotations_dir": "components",
                    "classes": self.config.component_classes,
                    "loss_weight": 1.0
                },
                "wire_detection": {
                    "type": "line_segment",
                    "annotations_dir": "wires",
                    "classes": self.config.wire_classes,
                    "loss_weight": 0.8
                },
                "node_detection": {
                    "type": "keypoint",
                    "annotations_dir": "nodes",
                    "classes": self.config.node_classes,
                    "loss_weight": 0.6
                },
                "terminal_detection": {
                    "type": "keypoint",
                    "annotations_dir": "terminals",
                    "loss_weight": 0.5
                }
            },
            "curriculum_learning": {
                "enabled": True,
                "ema_decay": 0.9,
                "difficulty_manifest": "difficulty_manifest.json",
                "stage_schedule": {
                    "stage_1_epochs": 5,
                    "stage_2_epochs": 10,
                    "stage_3_epochs": 15,
                    "stage_4_epochs": 20,
                    "stage_5_epochs": 25
                }
            }
        }
        
        with open(output_path / "multihead_config.json", 'w') as f:
            json.dump(config, f, indent=2)

