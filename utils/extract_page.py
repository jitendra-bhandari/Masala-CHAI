import os
import sys
import json
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

ANNOTATION_JSON = "annotation_data.json"
OUTPUT_DIR = "cropped_images"

# Maximum dimension for displaying the page in the annotation window.
DISPLAY_MAX_DIM = 1000

# Final size for saving cropped diagrams (e.g., 224 x 224).
CROP_SAVE_SIZE = (400, 295)

def load_annotation_data(json_file):
    """Load existing annotation data or return a fresh structure."""
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            return json.load(f)
    return {"pages": {}}

def save_annotation_data(json_file, data):
    """Save annotation data to the JSON file."""
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

def resize_with_aspect_ratio(img_cv2, max_dim=1600):
    """
    Resize an OpenCV image so that the larger dimension is at most `max_dim`,
    preserving aspect ratio. Return the resized image and the scale factor.
    """
    height, width = img_cv2.shape[:2]
    max_current_dim = max(width, height)
    if max_current_dim <= max_dim:
        # No resizing needed
        return img_cv2, 1.0

    scale = max_dim / float(max_current_dim)
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_img = cv2.resize(img_cv2, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_img, scale

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_page.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load or create JSON annotation data
    annotation_data = load_annotation_data(ANNOTATION_JSON)
    completed_pages = set(int(p) for p in annotation_data["pages"].keys())

    print("Converting PDF pages to images. This may take a moment...")
    # Convert PDF to a list of PIL Images (one per page)
    pages = convert_from_path(pdf_path)

    total_pages = len(pages)
    print(f"Total pages in PDF: {total_pages}")

    # Iterate over pages
    for page_index, pil_img in enumerate(pages):
        page_number = page_index + 1  # 1-based page numbering

        # Skip if this page is already annotated
        if page_number in completed_pages:
            print(f"Skipping page {page_number}, already annotated.")
            continue

        # Convert PIL image to OpenCV (NumPy) format
        cv_img_original = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        orig_height, orig_width = cv_img_original.shape[:2]

        # We'll store the original page size in the JSON
        page_size = [orig_width, orig_height]

        # Resize for ROI selection if necessary
        cv_img_display, scale = resize_with_aspect_ratio(cv_img_original, DISPLAY_MAX_DIM)

        # Show instructions in the console
        print(f"\n=== Page {page_number}/{total_pages} ===")
        print("Draw bounding boxes on the displayed image, press ENTER after each box.")
        print("Press ESC or 'Cancel' to finish selection (no boxes) for this page.")

        # selectROIs returns a list of [x, y, w, h] bounding boxes on the resized image
        rois = cv2.selectROIs(
            f"Select Diagrams (Page {page_number})",
            cv_img_display,
            showCrosshair=True,
            fromCenter=False
        )
        cv2.destroyAllWindows()  # close the ROI selection window

        # If rois is empty (user pressed ESC or canceled)
        if len(rois) == 0:
            print(f"No bounding boxes selected for page {page_number}. Marking as completed.")
            annotation_data["pages"][str(page_number)] = {
                "page_size": page_size,
                "images": [],
                "num_images": 0
            }
            completed_pages.add(page_number)
            save_annotation_data(ANNOTATION_JSON, annotation_data)
        else:
            print(f"User selected {len(rois)} bounding boxes for page {page_number}.")

            # Prepare to store data in JSON
            if str(page_number) not in annotation_data["pages"]:
                annotation_data["pages"][str(page_number)] = {
                    "page_size": page_size,
                    "images": []
                }

            image_list = annotation_data["pages"][str(page_number)]["images"]

            # Crop and save each ROI
            for idx, (x, y, w, h) in enumerate(rois, start=1):
                # Convert ROI from display scale to original scale
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(w / scale)
                h_orig = int(h / scale)

                left = x_orig
                upper = y_orig
                right = x_orig + w_orig
                lower = y_orig + h_orig

                # Crop using the original PIL image for best quality
                cropped_pil = pil_img.crop((left, upper, right, lower))
                # Optionally, save the cropped images at a uniform 224×224 resolution
                cropped_pil = cropped_pil.resize(CROP_SAVE_SIZE, Image.Resampling.LANCZOS)

                out_filename = f"{page_number}_{idx}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_filename)
                cropped_pil.save(out_path)

                # Store annotation info (in the ORIGINAL page space)
                bbox_info = {
                    "file": out_filename,
                    "coords": [left, upper, right, lower]
                }
                image_list.append(bbox_info)

            # Update JSON
            annotation_data["pages"][str(page_number)]["num_images"] = len(image_list)
            completed_pages.add(page_number)
            save_annotation_data(ANNOTATION_JSON, annotation_data)
            print(f"Saved {len(rois)} cropped images for page {page_number}.")

        # ---------------------------------------------------------------
        # After finishing this page, let user press 'S' to stop script
        # or press any other key to continue automatically to next page.
        info_window = np.zeros((100, 540, 3), dtype=np.uint8)
        cv2.putText(info_window,
                    "Press 'S' to stop now, or any other key to continue",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)
        cv2.imshow("Stop or Continue?", info_window)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("Stop or Continue?")
        if key == ord('s') or key == ord('S'):
            print("User chose to stop annotation. Exiting now.")
            break
        # ---------------------------------------------------------------

    print("\nAnnotation complete (or user chose to stop).")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
