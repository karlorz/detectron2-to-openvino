#!/usr/bin/env python3
"""
Real-time Object Detection using YOLO-NAS converted to OpenVINO
This script converts YOLO-NAS models to OpenVINO format and runs inference
"""

import cv2
import numpy as np
import torch
import openvino as ov
from pathlib import Path
import time
from typing import List, Dict, Tuple
import warnings

class YOLONASOpenVINODetector:
    def __init__(self, model_size: str = "yolo_nas_s"):
        """
        Initialize YOLO-NAS OpenVINO detector
        
        Args:
            model_size: One of 'yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'
        """
        self.model_size = model_size
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.input_size = (640, 640)
        
        self.ov_model = None
        self.compiled_model = None
        self.core = ov.Core()
        
        # COCO class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Generate colors for each class
        np.random.seed(42)  # For consistent colors
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))

    def convert_yolo_nas_to_openvino(self):
        """Convert YOLO-NAS PyTorch model to OpenVINO format"""
        try:
            from super_gradients.training import models
            print(f"Loading YOLO-NAS {self.model_size} model...")
            
            # Load pre-trained YOLO-NAS model
            model = models.get(self.model_size, pretrained_weights="coco")
            model.eval()
            
            print("Converting to OpenVINO format...")
            
            # Create dummy input for conversion
            dummy_input = torch.randn(1, 3, 640, 640)
            
            # Convert to OpenVINO
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ov_model = ov.convert_model(
                    model,
                    example_input=dummy_input,
                    input=[1, 3, 640, 640]
                )
            
            return ov_model
            
        except ImportError:
            print("super-gradients not installed. Install with:")
            print("uv add super-gradients")
            return None
        except Exception as e:
            print(f"Error converting model: {e}")
            return None

    def load_or_convert_model(self, device: str = "CPU"):
        """Load existing OpenVINO model or convert from YOLO-NAS"""
        model_xml_path = self.model_dir / f"{self.model_size}.xml"
        
        if model_xml_path.exists():
            print(f"Loading existing OpenVINO model from {model_xml_path}")
            self.ov_model = self.core.read_model(model_xml_path)
        else:
            print("OpenVINO model not found. Converting from YOLO-NAS...")
            
            # Convert model
            self.ov_model = self.convert_yolo_nas_to_openvino()
            
            if self.ov_model is None:
                return False
            
            # Save converted model
            print(f"Saving converted model to {model_xml_path}")
            ov.save_model(self.ov_model, str(model_xml_path))
        
        # Compile model for inference
        print(f"Compiling model for {device} device...")
        self.compiled_model = self.core.compile_model(self.ov_model, device)
        print("Model loaded and compiled successfully!")
        return True

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO-NAS inference"""
        # Resize image to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert HWC to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched

    def postprocess_detections(self, outputs: np.ndarray, original_shape: Tuple[int, int]) -> List[Dict]:
        """Post-process YOLO-NAS outputs to get final detections"""
        detections = []
        
        # YOLO-NAS output format: [batch, num_detections, 85]
        # where 85 = 4 (bbox) + 1 (confidence) + 80 (class scores)
        if len(outputs) > 0:
            predictions = outputs[0]  # Get first output
            
            # Handle different output formats
            if len(predictions.shape) == 3:
                predictions = predictions[0]  # Remove batch dimension
            
            # Extract boxes, confidences, and class scores
            boxes = predictions[:, :4]  # x1, y1, x2, y2
            confidences = predictions[:, 4]  # objectness score
            class_scores = predictions[:, 5:]  # class probabilities
            
            # Get class predictions
            class_ids = np.argmax(class_scores, axis=1)
            class_confidences = np.max(class_scores, axis=1)
            
            # Combine objectness and class confidence
            final_confidences = confidences * class_confidences
            
            # Filter by confidence threshold
            valid_detections = final_confidences >= self.confidence_threshold
            
            if np.any(valid_detections):
                boxes = boxes[valid_detections]
                final_confidences = final_confidences[valid_detections]
                class_ids = class_ids[valid_detections]
                
                # Scale boxes to original image size
                scale_x = original_shape[1] / self.input_size[0]
                scale_y = original_shape[0] / self.input_size[1]
                
                boxes[:, [0, 2]] *= scale_x  # x coordinates
                boxes[:, [1, 3]] *= scale_y  # y coordinates
                
                # Apply NMS
                indices = cv2.dnn.NMSBoxes(
                    boxes.tolist(),
                    final_confidences.tolist(),
                    self.confidence_threshold,
                    self.nms_threshold
                )
                
                if len(indices) > 0:
                    indices = indices.flatten()
                    
                    for i in indices:
                        x1, y1, x2, y2 = boxes[i]
                        confidence = final_confidences[i]
                        class_id = class_ids[i]
                        
                        # Convert to x, y, w, h format
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        
                        # Ensure coordinates are within image bounds
                        x = max(0, min(x, original_shape[1] - 1))
                        y = max(0, min(y, original_shape[0] - 1))
                        w = max(1, min(w, original_shape[1] - x))
                        h = max(1, min(h, original_shape[0] - y))
                        
                        detections.append({
                            'class_id': int(class_id),
                            'class_name': self.class_names[int(class_id)],
                            'confidence': float(confidence),
                            'bbox': [x, y, w, h]
                        })
        
        return detections

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run object detection on an image"""
        if self.compiled_model is None:
            return []
        
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.compiled_model(input_tensor)
        
        # Convert outputs to numpy arrays
        output_arrays = []
        for output in outputs.values():
            output_arrays.append(output)
        
        # Post-process outputs
        detections = self.postprocess_detections(output_arrays, image.shape[:2])
        
        return detections

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        result_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # Draw bounding box
            color = tuple(self.colors[class_id].astype(int).tolist())
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(result_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_image

def main():
    """Main function to run webcam detection demo"""
    print("Initializing YOLO-NAS OpenVINO Object Detector...")
    
    # Initialize detector (you can change to 'yolo_nas_m' or 'yolo_nas_l' for better accuracy)
    detector = YOLONASOpenVINODetector("yolo_nas_s")
    
    # Load or convert model
    if not detector.load_or_convert_model("CPU"):
        print("Failed to load model. Make sure super-gradients is installed:")
        print("uv add super-gradients")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting webcam detection. Press 'q' to quit.")
    
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Run detection
        detections = detector.detect(frame)
        
        # Print detections to console
        if detections:
            print(f"Frame {fps_counter}: Detected {len(detections)} objects:")
            for det in detections:
                print(f"  - {det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")
        
        # Draw results
        result_frame = detector.draw_detections(frame, detections)
        
        # Calculate and display FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = fps_counter / elapsed_time
            print(f"FPS: {fps:.2f}")
        
        # Display frame
        cv2.imshow('YOLO-NAS OpenVINO Object Detection', result_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    main()