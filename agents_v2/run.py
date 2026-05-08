#!/usr/bin/env python3
"""
Agents V2 Pipeline - CLI Entry Point with Parallel Processing

Usage:
    # Single image
    python -m agents_v2.run --image sample-images/330.jpg
    
    # Batch processing (sequential)
    python -m agents_v2.run --batch sample-images/ --output results/
    
    # Parallel batch processing (recommended for large batches)
    python -m agents_v2.run --batch sample-images/ --parallel --workers 4
    
    # Full scale (6500 images)
    python -m agents_v2.run --batch /path/to/images/ --parallel --workers 8 --batch-size 100
"""

import argparse
import json
import sys
import os
import shutil
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
# output_dir/
# ├── images/           # Original images (copied)
# │   ├── 330.jpg
# │   └── ...
# ├── spice/            # SPICE netlist files only
# │   ├── 330.sp
# │   └── ...
# ├── captions/         # Circuit captions/descriptions
# │   ├── 330.txt
# │   └── ...
# ├── detection/        # CV detection outputs per image
# │   ├── 330/
# │   │   ├── components_overlay.png
# │   │   ├── lines_overlay.png
# │   │   ├── detection_data.json
# │   │   └── ...
# │   └── ...
# ├── agents/           # Agent outputs per image
# │   ├── 330/
# │   │   ├── 330_result.json
# │   │   ├── 330_llm_flow.md
# │   │   ├── 330_simulation.log
# │   │   └── ...
# │   └── ...
# └── summary.json      # Batch summary
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BatchStats:
    """Statistics for batch processing"""
    total: int = 0
    processed: int = 0
    success: int = 0
    partial: int = 0
    failed: int = 0
    errors: int = 0
    needs_review: int = 0
    total_time: float = 0.0
    avg_time_per_image: float = 0.0
    total_tokens: int = 0
    total_llm_calls: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "total": self.total,
            "processed": self.processed,
            "success": self.success,
            "partial": self.partial,
            "failed": self.failed,
            "errors": self.errors,
            "needs_review": self.needs_review,
            "total_time_seconds": round(self.total_time, 2),
            "avg_time_per_image_seconds": round(self.avg_time_per_image, 2),
            "total_tokens": self.total_tokens,
            "total_llm_calls": self.total_llm_calls
        }


def setup_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create output directory structure"""
    dirs = {
        "root": output_dir,
        "images": output_dir / "images",
        "spice": output_dir / "spice",
        "captions": output_dir / "captions",
        "detection": output_dir / "detection",
        "agents": output_dir / "agents"
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return dirs


def load_cv_detections(
    image_path: str, 
    output_dirs: Dict[str, Path],
    image_stem: str,
    quiet: bool = False
) -> Dict:
    """
    Load or run CV detections for an image using the actual detection pipeline.
    Saves outputs to detection/{image_stem}/ directory.
    """
    import cv2
    import numpy as np
    
    if not quiet:
        print(f"  Running CV detection pipeline...")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    height, width = img.shape[:2]
    
    # Create per-image detection directory
    detection_dir = output_dirs["detection"] / image_stem
    detection_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import the main detection pipeline
        from main import main as run_component_detection
        from PIL import Image
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        
        # Load image as numpy array
        pil_img = Image.open(image_path)
        inp = np.array(pil_img)
        
        # Save scanned_circuit.png for detection pipeline
        scanned_path = detection_dir / 'scanned_circuit.png'
        try:
            # Use PIL to save directly instead of matplotlib for reliability
            pil_img.save(str(scanned_path))
        except Exception as e:
            # Fallback to matplotlib
            try:
                plt.figure(figsize=(5, 5))
                plt.imshow(pil_img)
                plt.axis('off')
                plt.savefig(str(scanned_path), bbox_inches='tight')
                plt.close()
            except Exception as e2:
                if not quiet:
                    print(f"    Warning: Could not save scanned_circuit.png: {e2}")
                # Last resort: copy original
                import shutil
                shutil.copy2(image_path, scanned_path)
        
        # Run the full detection pipeline
        rebuilt, comp, nodes, _, og_with_labels, comp_with_labels, comp_list, jns_list, conn_list, sample_dict = run_component_detection(
            str(detection_dir), inp, image_stem, use_clustering=False
        )
        
        # Get YOLO detections directly from the YOLO model
        from utils.yolov8 import comp_detection
        _, dim_matrix = comp_detection(str(scanned_path))
        
        # Classes from YOLO model (must match main.py exactly!)
        yolo_classes = ['AC_Source', 'BJT', 'Battery', 'Capacitor', 'Current_Source', 'DC_Source', 'Diode', 'Ground', 'Inductor', 'MOSFET', 'Resistor', 'Voltage_Source']
        
        # Save detection overlays with error handling
        try:
            if comp_with_labels is not None and hasattr(comp_with_labels, 'shape'):
                Image.fromarray(comp_with_labels).save(str(detection_dir / 'components_overlay.png'))
        except Exception as e:
            if not quiet:
                print(f"    Warning: Could not save components_overlay: {e}")
        
        try:
            if og_with_labels is not None and hasattr(og_with_labels, 'shape'):
                Image.fromarray(og_with_labels).save(str(detection_dir / 'lines_overlay.png'))
        except Exception as e:
            if not quiet:
                print(f"    Warning: Could not save lines_overlay: {e}")
        
        # Save optional visualizations with validation
        def safe_save_plot(img_array, filepath, title=""):
            """Safely save a plot from numpy array"""
            try:
                if img_array is None:
                    return False
                if not hasattr(img_array, 'shape'):
                    return False
                if len(img_array.shape) < 2:
                    return False
                # Ensure it's a valid image array
                if img_array.dtype != np.uint8:
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)
                
                plt.figure(figsize=(8, 8))
                plt.imshow(img_array)
                plt.axis('off')
                if title:
                    plt.title(title)
                plt.savefig(str(filepath), bbox_inches='tight', dpi=150)
                plt.close()
                return True
            except Exception as e:
                if not quiet:
                    print(f"    Warning: Could not save {filepath.name}: {e}")
                plt.close('all')  # Clean up any open figures
                return False
        
        safe_save_plot(comp, detection_dir / 'detected_components.png', 
                      f"Detected Components ({len(comp_list)})")
        safe_save_plot(nodes, detection_dir / 'nodes_terminals.png',
                      f"Nodes & Terminals ({len(jns_list)})")
        safe_save_plot(rebuilt, detection_dir / 'rebuilt_circuit.png',
                      "Rebuilt Circuit")
        
        # Save text descriptions
        with open(detection_dir / 'components.txt', 'w') as f:
            f.write(f"# Components Detected: {len(comp_list)}\n\n")
            for line in comp_list:
                f.write(f"{line}\n")
        
        with open(detection_dir / 'nodes.txt', 'w') as f:
            f.write(f"# Nodes Detected: {len(jns_list)}\n\n")
            for line in jns_list:
                f.write(f"{line}\n")
        
        with open(detection_dir / 'connections.txt', 'w') as f:
            f.write(f"# Connections: {len(conn_list)}\n\n")
            for line in conn_list:
                f.write(f"{line}\n")
        
        if not quiet:
            print(f"    Components: {len(comp_list)}, Nodes: {len(jns_list)}, Connections: {len(conn_list)}")
        
        # Parse YOLO detections directly from dim_matrix
        yolo_detections = []
        for i in range(dim_matrix.shape[0]):
            dim = dim_matrix[i]
            class_idx = int(dim[5])
            class_name = yolo_classes[class_idx] if class_idx < len(yolo_classes) else "Unknown"
            
            yolo_detections.append({
                "class": class_name,
                "confidence": float(dim[4]) if len(dim) > 4 else 0.8,
                "x_min": int(dim[0]),
                "y_min": int(dim[1]),
                "x_max": int(dim[2]),
                "y_max": int(dim[3])
            })
        
        # Parse Hough lines
        hough_lines = []
        if 'lines' in sample_dict:
            for line in sample_dict.get('lines', []):
                hough_lines.append({
                    "start": {"x": int(line[0]), "y": int(line[1])},
                    "end": {"x": int(line[2]), "y": int(line[3])}
                })
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=10)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    hough_lines.append({
                        "start": {"x": int(x1), "y": int(y1)},
                        "end": {"x": int(x2), "y": int(y2)}
                    })
        
        # Parse junction candidates
        junctions = []
        for jn_str in jns_list:
            try:
                import re
                coords = re.findall(r'\((\d+),\s*(\d+)\)', jn_str)
                if coords:
                    for x, y in coords:
                        junctions.append({"x": int(x), "y": int(y)})
            except Exception:
                pass
        
        if not junctions and 'junctions' in sample_dict:
            for jn in sample_dict.get('junctions', []):
                junctions.append({
                    "x": int(jn.get('x', jn[0] if isinstance(jn, list) else 0)),
                    "y": int(jn.get('y', jn[1] if isinstance(jn, list) else 0))
                })
        
        result = {
            "yolo_detections": yolo_detections,
            "hough_lines": hough_lines,
            "junction_candidates": junctions,
            "image_size": {"width": width, "height": height},
            "comp_list": comp_list,
            "jns_list": jns_list,
            "conn_list": conn_list,
            "sample_dict": sample_dict
        }
        
        # Save detection data as JSON
        detection_data = {
            "yolo_detections": yolo_detections,
            "hough_lines": hough_lines,
            "junction_candidates": junctions,
            "image_size": {"width": width, "height": height},
            "components": comp_list,
            "nodes": jns_list,
            "connections": conn_list
        }
        with open(detection_dir / 'detection_data.json', 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        return result
        
    except ImportError as e:
        if not quiet:
            print(f"  Warning: Main detection pipeline not available ({e})")
            print(f"  Falling back to direct YOLO + Hough detection...")
        
        try:
            from ultralytics import YOLO
            
            yolo_path = "trained_checkpoints/yolov8_best.pt"
            if not Path(yolo_path).exists():
                yolo_path = "runs/train/circuit_yolov11s_stable/weights/best.pt"
            
            yolo_model = YOLO(yolo_path)
            yolo_results = yolo_model(image_path, conf=0.25)
            
            yolo_detections = []
            for result in yolo_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    yolo_detections.append({
                        "class": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "x_min": int(x1),
                        "y_min": int(y1),
                        "x_max": int(x2),
                        "y_max": int(y2)
                    })
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=10)
            
            hough_lines = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    hough_lines.append({
                        "start": {"x": int(x1), "y": int(y1)},
                        "end": {"x": int(x2), "y": int(y2)}
                    })
            
            from utils.node_detector import node_detector
            junctions_raw = node_detector(gray)
            junctions = [{"x": int(j[1]), "y": int(j[0])} for j in junctions_raw]
            
            return {
                "yolo_detections": yolo_detections,
                "hough_lines": hough_lines,
                "junction_candidates": junctions,
                "image_size": {"width": width, "height": height}
            }
            
        except Exception as e2:
            return {
                "yolo_detections": [],
                "hough_lines": [],
                "junction_candidates": [],
                "image_size": {"width": width, "height": height},
                "error": str(e2)
            }


def process_single_image(
    image_path: str,
    output_dirs: Dict[str, Path],
    config: 'PipelineConfig',
    quiet: bool = False
) -> Tuple[str, Dict, Optional[str]]:
    """
    Process a single image through the full pipeline.
    Returns: (image_stem, output_dict, error_message)
    """
    from agents_v2.workflow import run_pipeline

    # Validate image_path
    if not isinstance(image_path, (str, Path)):
        return ("invalid", {
            "status": "ERROR",
            "confidence": 0.0,
            "needs_review": True,
            "review_reasons": [f"Invalid image path type: {type(image_path)}"],
            "tokens": 0,
            "llm_calls": 0,
            "time": 0.0
        }, f"Invalid image path type: {type(image_path)}")

    image_stem = Path(image_path).stem
    start_time = time.time()

    try:
        # Copy original image to images/
        dest_image = output_dirs["images"] / Path(image_path).name
        if not dest_image.exists():
            shutil.copy2(image_path, dest_image)
        
        # Run CV detection
        detections = load_cv_detections(image_path, output_dirs, image_stem, quiet)
        
        # Create per-image agents directory
        agents_dir = output_dirs["agents"] / image_stem
        agents_dir.mkdir(parents=True, exist_ok=True)
        
        # Run agent pipeline
        output = run_pipeline(
            image_path=image_path,
            yolo_detections=detections["yolo_detections"],
            hough_lines=detections["hough_lines"],
            junction_candidates=detections["junction_candidates"],
            config=config,
            output_subdir=str(agents_dir)
        )
        
        # Copy .sp to spice/ directory
        sp_file = agents_dir / f"{image_stem}.sp"
        if sp_file.exists():
            shutil.copy2(sp_file, output_dirs["spice"] / f"{image_stem}.sp")
        
        # Copy caption to captions/ directory
        caption_file = agents_dir / f"{image_stem}_caption.txt"
        if caption_file.exists():
            shutil.copy2(caption_file, output_dirs["captions"] / f"{image_stem}.txt")
        
        elapsed = time.time() - start_time
        
        return (image_stem, {
            "status": output.overall_status,
            "confidence": output.overall_confidence,
            "needs_review": output.needs_review,
            "review_reasons": output.review_reasons[:3] if output.review_reasons else [],
            "tokens": output.total_tokens,
            "llm_calls": output.total_llm_calls,
            "time": round(elapsed, 2)
        }, None)
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        if not quiet:
            print(f"  ERROR: {error_msg}")
            traceback.print_exc()
        
        return (image_stem, {
            "status": "ERROR",
            "confidence": 0.0,
            "needs_review": True,
            "review_reasons": [error_msg],
            "tokens": 0,
            "llm_calls": 0,
            "time": round(elapsed, 2)
        }, error_msg)


def worker_init():
    """Initialize worker process"""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set matplotlib backend for subprocess
    import matplotlib
    matplotlib.use('Agg', force=True)


def process_batch_parallel(
    images: List[Path],
    output_dirs: Dict[str, Path],
    config: 'PipelineConfig',
    num_workers: int = 4,
    batch_size: int = 50,
    quiet: bool = False
) -> Tuple[List[Dict], BatchStats]:
    """
    Process images in parallel using ProcessPoolExecutor.
    
    For CV detection (CPU-bound): use ProcessPoolExecutor
    For LLM calls (IO-bound): could use ThreadPoolExecutor
    
    Due to complexity of mixed workload, we use ThreadPoolExecutor
    which works better with shared resources.
    """
    stats = BatchStats(total=len(images))
    results = []
    
    # Process in batches to manage memory
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    print(f"\n{'='*70}")
    print(f"  PARALLEL BATCH PROCESSING")
    print(f"  Total images: {len(images)}")
    print(f"  Workers: {num_workers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {num_batches}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(images))
        batch_images = images[batch_start:batch_end]
        
        print(f"\n  Batch {batch_idx + 1}/{num_batches} ({len(batch_images)} images)")
        print(f"  {'-'*60}")
        
        batch_start_time = time.time()
        
        # Use ThreadPoolExecutor for mixed IO/CPU workload
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_single_image,
                    str(img_path),
                    output_dirs,
                    config,
                    quiet=True  # Quiet in parallel mode
                ): img_path
                for img_path in batch_images
            }
            
            for future in as_completed(futures):
                img_path = futures[future]
                try:
                    image_stem, result, error = future.result()
                    results.append({"image": image_stem, **result})
                    
                    stats.processed += 1
                    stats.total_tokens += result.get("tokens", 0)
                    stats.total_llm_calls += result.get("llm_calls", 0)
                    
                    if error:
                        stats.errors += 1
                        # Show more of the error message for debugging
                        error_display = error if len(error) <= 100 else f"{error[:100]}..."
                        print(f"    ✗ {image_stem}: ERROR - {error_display}")
                    elif result["status"] == "SUCCESS":
                        stats.success += 1
                        print(f"    ✓ {image_stem}: SUCCESS ({result['confidence']:.2f})")
                    elif result["status"] == "PARTIAL":
                        stats.partial += 1
                        print(f"    ⚠ {image_stem}: PARTIAL ({result['confidence']:.2f})")
                    else:
                        stats.failed += 1
                        print(f"    ✗ {image_stem}: FAILED")
                    
                    if result.get("needs_review"):
                        stats.needs_review += 1
                        
                except Exception as e:
                    stats.errors += 1
                    img_name = img_path.stem if hasattr(img_path, 'stem') else str(img_path)
                    print(f"    ✗ {img_name}: EXCEPTION - {e}")
        
        batch_time = time.time() - batch_start_time
        print(f"  Batch completed in {batch_time:.1f}s ({batch_time/len(batch_images):.1f}s/image)")
    
    stats.total_time = time.time() - start_time
    stats.avg_time_per_image = stats.total_time / max(stats.processed, 1)
    
    return results, stats


def process_batch_sequential(
    images: List[Path],
    output_dirs: Dict[str, Path],
    config: 'PipelineConfig',
    quiet: bool = False
) -> Tuple[List[Dict], BatchStats]:
    """Process images sequentially (for debugging or small batches)"""
    stats = BatchStats(total=len(images))
    results = []
    
    print(f"\n{'='*70}")
    print(f"  SEQUENTIAL BATCH PROCESSING")
    print(f"  Total images: {len(images)}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] {img_path.name}")
        
        image_stem, result, error = process_single_image(
            str(img_path),
            output_dirs,
            config,
            quiet
        )
        
        results.append({"image": image_stem, **result})
        
        stats.processed += 1
        stats.total_tokens += result.get("tokens", 0)
        stats.total_llm_calls += result.get("llm_calls", 0)
        
        if error:
            stats.errors += 1
        elif result["status"] == "SUCCESS":
            stats.success += 1
        elif result["status"] == "PARTIAL":
            stats.partial += 1
        else:
            stats.failed += 1
        
        if result.get("needs_review"):
            stats.needs_review += 1
        
        print(f"  Result: {result['status']} (conf: {result['confidence']:.2f}, time: {result['time']:.1f}s)")
    
    stats.total_time = time.time() - start_time
    stats.avg_time_per_image = stats.total_time / max(stats.processed, 1)
    
    return results, stats


def run_single_image(
    image_path: str,
    config: 'PipelineConfig',
    output_dir: str = "agents_v2_output"
) -> Dict:
    """Run pipeline on a single image"""
    
    print(f"\n{'='*70}")
    print(f"  Processing: {image_path}")
    print(f"{'='*70}")
    
    # Setup output directories
    output_dirs = setup_output_dirs(Path(output_dir))
    
    # Process image
    image_stem, result, error = process_single_image(
        image_path,
        output_dirs,
        config,
        quiet=False
    )
    
    if error:
        print(f"\n  ERROR: {error}")
    else:
        print(f"\n  Result: {result['status']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        if result.get('needs_review'):
            print(f"  ⚠ NEEDS REVIEW: {result['review_reasons']}")
    
    # Create a minimal output object for compatibility
    from types import SimpleNamespace
    output = SimpleNamespace(
        overall_status=result['status'],
        overall_confidence=result['confidence'],
        needs_review=result.get('needs_review', False),
        review_reasons=result.get('review_reasons', []),
        total_tokens=result.get('tokens', 0),
        total_llm_calls=result.get('llm_calls', 0)
    )
    
    return output


def run_batch(
    image_dir: str,
    config: 'PipelineConfig',
    output_dir: str,
    parallel: bool = False,
    num_workers: int = 4,
    batch_size: int = 50,
    export_training: bool = False
) -> Tuple[List[Dict], BatchStats]:
    """Run pipeline on batch of images"""
    
    image_dir = Path(image_dir)
    output_dirs = setup_output_dirs(Path(output_dir))
    
    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = sorted([
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    
    if not images:
        print(f"No images found in {image_dir}")
        return [], BatchStats()
    
    # Process
    if parallel:
        results, stats = process_batch_parallel(
            images, output_dirs, config, num_workers, batch_size
        )
    else:
        results, stats = process_batch_sequential(
            images, output_dirs, config
        )
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": str(image_dir),
        "output_dir": str(output_dir),
        "parallel": parallel,
        "workers": num_workers if parallel else 1,
        "batch_size": batch_size if parallel else 1,
        "stats": stats.to_dict(),
        "results": results
    }
    
    summary_path = output_dirs["root"] / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  BATCH SUMMARY")
    print(f"{'='*70}")
    print(f"  Total:        {stats.total}")
    print(f"  Processed:    {stats.processed}")
    print(f"  Success:      {stats.success} ({100*stats.success/max(stats.processed,1):.1f}%)")
    print(f"  Partial:      {stats.partial} ({100*stats.partial/max(stats.processed,1):.1f}%)")
    print(f"  Failed:       {stats.failed} ({100*stats.failed/max(stats.processed,1):.1f}%)")
    print(f"  Errors:       {stats.errors}")
    print(f"  Needs Review: {stats.needs_review}")
    print(f"  {'-'*60}")
    print(f"  Total Time:   {stats.total_time:.1f}s ({stats.total_time/60:.1f}m)")
    print(f"  Avg/Image:    {stats.avg_time_per_image:.1f}s")
    print(f"  LLM Calls:    {stats.total_llm_calls}")
    print(f"  Total Tokens: {stats.total_tokens}")
    print(f"  {'-'*60}")
    print(f"  Summary saved: {summary_path}")
    print(f"{'='*70}")
    
    return results, stats


def main():
    parser = argparse.ArgumentParser(
        description="Agents V2 Refinement Pipeline with Parallel Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python -m agents_v2.run --image sample-images/330.jpg
  
  # Sequential batch
  python -m agents_v2.run --batch sample-images/ --output results/
  
  # Parallel batch (recommended for large datasets)
  python -m agents_v2.run --batch sample-images/ --parallel --workers 4
  
  # Large scale (6500 images)
  python -m agents_v2.run --batch /path/to/images/ --parallel --workers 8 --batch-size 100
  
  # With custom models
  python -m agents_v2.run --batch images/ --parallel --agent-model gpt-4o
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", "-i", help="Single image to process")
    input_group.add_argument("--batch", "-b", help="Directory of images to process")
    
    # Output options
    parser.add_argument("--output", "-o", default="agents_v2_output",
                       help="Output directory (default: agents_v2_output)")
    parser.add_argument("--export-training", action="store_true",
                       help="Export training data for curriculum learning")
    
    # Parallel processing options
    parser.add_argument("--parallel", "-p", action="store_true",
                       help="Enable parallel processing")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Images per batch for memory management (default: 50)")
    
    # Model options
    parser.add_argument("--agent-model", default="gpt-5.2",
                       help="Model for agents 1-3 (default: gpt-5.2)")
    parser.add_argument("--judge-model", default="gpt-5.2",
                       help="Model for LLM-as-Judge (default: gpt-5.2)")
    parser.add_argument("--use-gpt4", action="store_true",
                       help="Use GPT-4o models instead of GPT-5.2")
    
    # Pipeline options
    parser.add_argument("--max-iterations", type=int, default=3,
                       help="Max iterations per agent (default: 3)")
    parser.add_argument("--pass-threshold", type=float, default=0.90,
                       help="Score threshold to pass (default: 0.90)")
    
    # Other options
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Import config here to avoid circular imports
    from agents_v2.config import PipelineConfig, ModelConfig
    
    # Build config
    if args.use_gpt4:
        model_config = ModelConfig(
            agent1_model="gpt-4o",
            agent2_model="gpt-4o",
            agent3_model="gpt-4o",
            judge_model="gpt-4o",
            use_gpt5=False
        )
    else:
        model_config = ModelConfig(
            agent1_model=args.agent_model,
            agent2_model=args.agent_model,
            agent3_model=args.agent_model,
            judge_model=args.judge_model,
            use_gpt5=True
        )
    
    config = PipelineConfig(
        models=model_config,
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    # Update thresholds if specified
    if args.pass_threshold:
        config.thresholds.agent1_pass_threshold = args.pass_threshold
        config.thresholds.agent2_pass_threshold = args.pass_threshold
        config.thresholds.agent3_pass_threshold = args.pass_threshold
    
    if args.max_iterations:
        config.iterations.agent1_max_iterations = args.max_iterations
        config.iterations.agent2_max_iterations = args.max_iterations
        config.iterations.agent3_max_iterations = args.max_iterations
    
    # Run
    start_time = time.time()
    
    if args.image:
        output = run_single_image(args.image, config, args.output)
    else:
        results, stats = run_batch(
            args.batch,
            config,
            args.output,
            parallel=args.parallel,
            num_workers=args.workers,
            batch_size=args.batch_size,
            export_training=args.export_training
        )
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}m)")


if __name__ == "__main__":
    main()
