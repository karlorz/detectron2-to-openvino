#!/usr/bin/env python3
"""
SSDLite MobileDet OpenVINO Detector
Downloads TensorFlow Lite model and converts to OpenVINO format for inference
"""

import cv2
import numpy as np
import openvino as ov
from pathlib import Path
import time
import os
import sys
import argparse
import requests
from typing import List, Dict, Tuple
import warnings

class SSDLiteMobileDetOpenVINODetector:
    def __init__(self, debug: bool = False):
        """
        Initialize SSDLite MobileDet OpenVINO detector
        
        Args:
            debug: Enable debug output
        """
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.debug = debug
        
        self.confidence_threshold = 0.3
        self.nms_threshold = 0.4
        self.input_size = (320, 320)  # SSDLite MobileDet input size
        
        self.ov_model = None
        self.compiled_model = None
        self.core = ov.Core()
        
        # COCO class names (91 classes including background)
        self.class_names = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Classes you want to detect (filter)
        self.wanted_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck'
        ]
        
        # Create mapping from class ID to class name and filter set
        self.wanted_class_ids = set()
        for class_name in self.wanted_classes:
            if class_name in self.class_names:
                class_id = self.class_names.index(class_name)
                self.wanted_class_ids.add(class_id)
        
        if debug:
            print(f"Filtering to detect only: {self.wanted_classes}")
            print(f"Class IDs to detect: {sorted(self.wanted_class_ids)}")
        
        # Generate colors for each class
        np.random.seed(42)
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        
        # Model URL for downloading TFLite model
        self.tflite_url = "https://github.com/google-coral/test_data/raw/release-frogfish/ssdlite_mobiledet_coco_qat_postprocess.tflite"

    def get_tflite_model_path(self) -> Path:
        """Get the path to the TFLite model, checking multiple locations"""
        # Check local models directory first
        local_path = self.model_dir / "ssdlite_mobiledet_cpu.tflite"
        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path
        
        # Check if running in Frigate container with pre-downloaded model
        frigate_path = Path("/opt/frigate/cpu_model.tflite")
        if frigate_path.exists() and frigate_path.stat().st_size > 0:
            print(f"Using pre-downloaded TFLite model from Frigate: {frigate_path}")
            return frigate_path
        
        # Check current directory
        current_path = Path("cpu_model.tflite")
        if current_path.exists() and current_path.stat().st_size > 0:
            print(f"Using TFLite model from current directory: {current_path}")
            return current_path
        
        return local_path  # Return local path for download

    def download_tflite_model(self) -> bool:
        """Download TensorFlow Lite model if not available locally"""
        tflite_path = self.get_tflite_model_path()
        
        # If model already exists, no need to download
        if tflite_path.exists() and tflite_path.stat().st_size > 0:
            print(f"TFLite model already available at: {tflite_path}")
            return True
        elif tflite_path.exists():
            print(f"TFLite model {tflite_path} exists but is empty, re-downloading...")
            tflite_path.unlink()
        
        print(f"Downloading SSDLite MobileDet TFLite model...")
        print(f"URL: {self.tflite_url}")
        
        try:
            response = requests.get(self.tflite_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(tflite_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}%", end='', flush=True)
            
            print(f"\nTFLite model downloaded successfully: {tflite_path} ({downloaded} bytes)")
            
            # Verify file was downloaded properly
            if tflite_path.stat().st_size == 0:
                raise Exception("Downloaded file is empty")
                
            return True
            
        except Exception as e:
            print(f"\nError downloading TFLite model: {e}")
            if tflite_path.exists():
                tflite_path.unlink()
            return False

    def convert_tflite_to_openvino(self) -> bool:
        """Convert TensorFlow Lite model to OpenVINO format"""
        tflite_path = self.get_tflite_model_path()
        xml_path = self.model_dir / "ssdlite_mobiledet.xml"
        
        if not tflite_path.exists():
            print(f"TensorFlow Lite model not found: {tflite_path}")
            return False
            
        try:
            print(f"Converting TensorFlow Lite model to OpenVINO format...")
            
            # Load and convert TFLite model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ov_model = ov.convert_model(str(tflite_path))
            
            # Save OpenVINO model
            print(f"Saving OpenVINO model to {xml_path}")
            ov.save_model(ov_model, str(xml_path))
            
            self.ov_model = ov_model
            print("Model conversion completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error converting TFLite to OpenVINO: {e}")
            return False

    def get_best_device(self) -> str:
        """Detect the best available device"""
        available_devices = self.core.available_devices
        print(f"Available devices: {available_devices}")
        
        # Check for GPU devices
        for device in available_devices:
            if "GPU" in device:
                print(f"Using {device} device for inference")
                return device
                
        print("Using CPU device for inference")
        return "CPU"

    def load_or_convert_model(self, device: str = None):
        """Load existing OpenVINO model or convert from TFLite"""
        model_xml_path = self.model_dir / "ssdlite_mobiledet.xml"
        
        # Auto-detect best device if not specified
        if device is None:
            device = self.get_best_device()
        
        # Try to load existing OpenVINO model first
        if model_xml_path.exists() and (self.model_dir / "ssdlite_mobiledet.bin").exists():
            try:
                print(f"Loading existing OpenVINO model from {model_xml_path}")
                self.ov_model = self.core.read_model(str(model_xml_path))
            except Exception as e:
                print(f"Error loading existing model: {e}")
                print("Will convert a new model...")
                self.ov_model = None
        
        # If no existing model, download and convert from TFLite
        if self.ov_model is None:
            print("OpenVINO model not found. Downloading TFLite model...")
            
            # Download TFLite model
            if not self.download_tflite_model():
                print("Failed to download TFLite model")
                return False
            
            # Convert TFLite to OpenVINO
            if not self.convert_tflite_to_openvino():
                print("Failed to convert TFLite model to OpenVINO")
                return False
        
        # Compile model for inference
        try:
            print(f"Compiling model for {device} device...")
            self.compiled_model = self.core.compile_model(self.ov_model, device)
            
            # Debug: Print model input/output info
            if self.debug:
                print("Model input information:")
                for input_layer in self.ov_model.inputs:
                    print(f"  Input: {input_layer.any_name}, shape: {input_layer.shape}, type: {input_layer.element_type}")
                print("Model output information:")
                for output_layer in self.ov_model.outputs:
                    print(f"  Output: {output_layer.any_name}, shape: {output_layer.shape}, type: {output_layer.element_type}")
            
            print("Model loaded and compiled successfully!")
            return True
        except Exception as e:
            print(f"Error compiling model for {device}: {e}")
            if device != "CPU":
                print("Falling back to CPU...")
                try:
                    self.compiled_model = self.core.compile_model(self.ov_model, "CPU")
                    print("Model compiled successfully on CPU!")
                    return True
                except Exception as cpu_e:
                    print(f"CPU fallback also failed: {cpu_e}")
            return False

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for SSDLite MobileDet inference"""
        # Resize to model input size
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched

    def postprocess_detections(self, outputs: List[np.ndarray], original_shape: Tuple[int, int]) -> List[Dict]:
        """Post-process SSDLite MobileDet outputs to get final detections"""
        detections = []
        
        if not outputs or len(outputs) < 4:
            print(f"Expected 4 outputs, got {len(outputs)}")
            return detections
        
        # SSDLite MobileDet output format:
        # Output 0: detection_boxes (1, 10, 4) - bounding boxes in [y1, x1, y2, x2] format
        # Output 1: detection_classes (1, 10) - class IDs
        # Output 2: detection_scores (1, 10) - confidence scores
        # Output 3: num_detections (1,) - number of valid detections
        
        try:
            boxes = outputs[0][0]  # Shape: (10, 4)
            classes = outputs[1][0]  # Shape: (10,)
            scores = outputs[2][0]  # Shape: (10,)
            num_detections = int(outputs[3][0])  # Scalar
            
            if self.debug:
                print(f"Number of detections: {num_detections}")
                print(f"Boxes shape: {boxes.shape}")
                print(f"Classes shape: {classes.shape}")
                print(f"Scores shape: {scores.shape}")
                print(f"Score range: {np.min(scores):.3f} - {np.max(scores):.3f}")
            
            # Use only the valid detections
            num_detections = min(num_detections, len(boxes))
            
            img_h, img_w = original_shape
            
            for i in range(num_detections):
                score = scores[i]
                class_id = int(classes[i])
                
                # Filter by confidence threshold and wanted classes
                if score < self.confidence_threshold:
                    continue
                    
                if class_id not in self.wanted_class_ids:
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                # TFLite model outputs normalized coordinates [y1, x1, y2, x2]
                y1, x1, y2, x2 = boxes[i]
                
                x1 = int(x1 * img_w)
                y1 = int(y1 * img_h)
                x2 = int(x2 * img_w)
                y2 = int(y2 * img_h)
                
                # Convert to [x, y, w, h] format
                x = max(0, x1)
                y = max(0, y1)
                w = max(1, min(x2 - x1, img_w - x))
                h = max(1, min(y2 - y1, img_h - y))
                
                # Ensure class_id is within bounds
                if 0 <= class_id < len(self.class_names):
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'confidence': float(score),
                        'bbox': [x, y, w, h]
                    })
                    
        except Exception as e:
            print(f"Error processing detections: {e}")
            return detections
        
        return detections

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run object detection on an image"""
        if self.compiled_model is None:
            return []
        
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.compiled_model(input_tensor)
        
        # Convert outputs to list
        output_arrays = []
        for key in sorted(outputs.keys()):
            output_arrays.append(outputs[key])
        
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
    parser = argparse.ArgumentParser(description='SSDLite MobileDet OpenVINO Object Detector')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--source', '-s', type=str, default='0', 
                       help='Video source: 0 for webcam, URL for stream, path for video file (default: 0)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SSDLite MobileDet OpenVINO Object Detector")
    print("=" * 60)
    
    detector = SSDLiteMobileDetOpenVINODetector(debug=args.debug)
    detector.confidence_threshold = args.threshold
    
    # Load model
    if not detector.load_or_convert_model():
        print("❌ Failed to load model.")
        return
    
    # Initialize video source
    def parse_source(source_str):
        try:
            return int(source_str)
        except ValueError:
            return source_str
    
    video_source = parse_source(args.source)
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source")
        return
    
    print("✅ Video source initialized successfully!")
    print("Starting real-time detection...")
    print("Press 'q' to quit")
    
    fps_counter = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Run detection
            detections = detector.detect(frame)
            
            # Print detections
            if detections and args.debug:
                print(f"Frame {fps_counter}: Detected {len(detections)} objects:")
                for det in detections:
                    print(f"  - {det['class_name']}: {det['confidence']:.2f}")
            
            # Draw results
            result_frame = detector.draw_detections(frame, detections)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = fps_counter / elapsed_time
                print(f"FPS: {fps:.2f}")
            
            # Display frame
            cv2.imshow('SSDLite MobileDet OpenVINO Detection', result_frame)
    
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")

if __name__ == "__main__":
    main()