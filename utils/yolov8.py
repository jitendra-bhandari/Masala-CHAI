from ultralytics import YOLO
import cv2
import os
import argparse

import cv2
import numpy as np

def annotate_bounding_boxes(image_path, predictions):
    """
    Annotates an image with bounding boxes and labels based on the given predictions tensor.
    
    Parameters:
    - image_path: Path to the input image.
    - predictions: A tensor containing bounding box coordinates, confidence scores, and class labels.

    Returns:
    - annotated_image: The image annotated with bounding boxes and labels as a numpy ndarray.
    - final_predictions: List of bounding box coordinates, scores, and labels.
    """
    
    def calculate_iou(box1, box2):
        """Calculate the Intersection over Union (IoU) of two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate intersection area
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate areas of each box
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate union area
        union_area = box1_area + box2_area - inter_area

        # Avoid division by zero
        if union_area == 0:
            return 0

        # Calculate IoU
        iou = inter_area / union_area
        return iou

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    # Extract classes and names for label mapping
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    names = ['AC_Source', 'BJT', 'Battery', 'Capacitor', 'Current_Source', 'DC_Source', 
             'Diode', 'Ground', 'Inductor', 'MOSFET', 'Resistor', 'Voltage_Source']
    
    final_predictions = []
    print("Shape of image",image.shape)
    
    # Iterate over the predictions
    for i, prediction in enumerate(predictions):
        x1, y1, x2, y2, score, class_id = prediction
        class_id = int(class_id)
        label = names[class_id]

        # Convert bounding box coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Apply rules for annotating capacitor
        if label == 'Capacitor':
            if score < 0.7:
                continue  # Skip if the confidence score is less than 0.7

            # Check overlap with MOSFETs
            overlap_with_mosfet = False
            for j, other_pred in enumerate(predictions):
                if i == j:
                    continue  # Skip self-comparison
                _, _, _, _, _, other_class_id = other_pred
                if int(other_class_id) == 9:  # Check only MOSFET class
                    other_x1, other_y1, other_x2, other_y2 = map(int, other_pred[:4])
                    iou = calculate_iou([x1, y1, x2, y2], [other_x1, other_y1, other_x2, other_y2])
                    if iou > 0.3:
                        overlap_with_mosfet = True
                        break

            if overlap_with_mosfet:
                continue  # Skip if overlap with MOSFET is greater than 0.3

        # Add to final predictions list
        final_predictions.append([x1, y1, x2, y2, score, class_id])

        # Annotate the image with bounding box and label
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        print("Coordinates from annotated_img:")
        print(x1,y1,x2,y2)
        text = f'{label} ({score:.2f})'
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert the final_predictions list to the specified format
    final_predictions = np.array(final_predictions, dtype=object)

    return image, final_predictions


def comp_detection(image_path):

    # Load the pretrained YOLO model
    model = YOLO("trained_checkpoints/yolov8_best.pt")

    if not os.path.exists(image_path):
            print(f"Error: The image {image_path} does not exist.")
    try:
        # Perform prediction
        result = model.predict(source=image_path)

        annotated_img,prediction = annotate_bounding_boxes(image_path,result[0].boxes.data)

        return annotated_img,prediction
            
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
