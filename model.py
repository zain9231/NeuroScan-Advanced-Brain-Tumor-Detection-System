# Import necessary libraries
import os
from ultralytics import YOLO
import cv2
import torch
import numpy as np

class TumorDetector:
    def __init__(self, model_path='C:\\neuro_scan_project\\models\\best.pt'):
        """
        Initialize the TumorDetector class.
        
        Args:
            model_path (str): Path to the pre-trained YOLO model file.
        """
        try:
            self.model = YOLO(model_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {str(e)}")

    def detect_tumor(self, image, confidence_threshold=0.5):
        """
        Perform tumor detection on the given image.
        
        Args:
            image: Input image for tumor detection.
            confidence_threshold (float): Minimum confidence score for a detection to be considered valid.
        
        Returns:
            numpy.ndarray: Array of detection results.
        """
        try:
            results = self.model(image, verbose=False)
            
            # Filter detections based on confidence threshold
            detections = results[0].boxes.data.cpu().numpy()
            filtered_detections = detections[detections[:, 4] >= confidence_threshold]
            
            return filtered_detections
        except Exception as e:
            print(f"Error during tumor detection: {str(e)}")
            return np.array([])

    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels for detected tumors on the image.
        
        Args:
            image: Original image.
            detections: Array of detection results.
        
        Returns:
            numpy.ndarray: Image with drawn detections.
        """
        image_copy = image.copy()
        for det in detections:
            try:
                bbox = det[:4].astype(int)
                conf = det[4]
                cls = int(det[5])
                label = f"{self.model.names[cls]} {conf:.2f}"
                cv2.rectangle(image_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(image_copy, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error drawing detection: {str(e)}")
        return image_copy

    def get_class_names(self):
        """
        Get the names of classes that the model can detect.
        
        Returns:
            dict: Dictionary of class indices and their corresponding names.
        """
        return self.model.names

    def format_detections(self, detections):
        """
        Format the detection results in a more informative way.
        
        Args:
            detections: Array of detection results.
        
        Returns:
            list: List of formatted detection results.
        """
        formatted_results = []
        for i, det in enumerate(detections, 1):
            bbox = det[:4].astype(int)
            conf = det[4]
            cls = int(det[5])
            class_name = self.model.names[cls]
            
            result = {
                "Detection Number": i,
                "Tumor Type": class_name,
                "Confidence": f"{conf:.2f}",
                "Bounding Box": bbox.tolist(),
                "Location": {
                    "Top-Left": (bbox[0], bbox[1]),
                    "Bottom-Right": (bbox[2], bbox[3])
                },
                "Size": {
                    "Width": bbox[2] - bbox[0],
                    "Height": bbox[3] - bbox[1]
                }
            }
            formatted_results.append(result)
        
        return formatted_results

    def format_detection_result(self, detection, image_shape):
        tumor_type = "Brain Tumor"  
        confidence = detection['confidence']
        top_left = detection['bbox'][:2]
        bottom_right = detection['bbox'][2:]
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        
        area = width * height
        total_area = image_shape[0] * image_shape[1]
        relative_size = (area / total_area) * 100

        center_x = (top_left[0] + bottom_right[0]) / 2
        center_y = (top_left[1] + bottom_right[1]) / 2
        position = ""
        if center_y < image_shape[0] / 2:
            position += "Upper "
        else:
            position += "Lower "
        if center_x < image_shape[1] / 2:
            position += "Left"
        else:
            position += "Right"

        return f"""
Detection {detection['detection_number']}:
  Tumor Type: {tumor_type}
  Confidence: {confidence:.2%}
  Location: Top-Left ({top_left[0]:.0f}, {top_left[1]:.0f}), Bottom-Right ({bottom_right[0]:.0f}, {bottom_right[1]:.0f}) pixels
  Size: {width:.0f} x {height:.0f} pixels ({area:.0f} sq. pixels)
  Relative Size: {relative_size:.2f}% of total image
  Position: {position} quadrant of the image
"""