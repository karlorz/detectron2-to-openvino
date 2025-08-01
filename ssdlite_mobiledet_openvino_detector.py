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
        
        self.confidence_threshold = 0.1
        self.nms_threshold = 0.4
        self.input_size = (300, 300)  # SSD MobileNet input size
        
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
        
        # Classes you want to detect (filter) - expanded to include common objects
        self.wanted_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'book', 'clock', 'vase'
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
        
        # Use pre-converted OpenVINO model from Intel's model zoo
        self.model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/ssd_mobilenet_v1_coco/FP32/ssd_mobilenet_v1_coco.xml"
        self.model_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/ssd_mobilenet_v1_coco/FP32/ssd_mobilenet_v1_coco.bin"
        self.model_format = "openvino"

    def get_model_paths(self) -> tuple[Path, Path]:
        """Get the paths to the OpenVINO model files (.xml and .bin)"""
        # Try the existing models first
        xml_path = self.model_dir / "ssd_mobilenet_v1.xml"
        bin_path = self.model_dir / "ssd_mobilenet_v1.bin"
        
        if xml_path.exists() and bin_path.exists():
            return xml_path, bin_path
        
        # Fallback to COCO version
        xml_path = self.model_dir / "ssd_mobilenet_v1_coco.xml"
        bin_path = self.model_dir / "ssd_mobilenet_v1_coco.bin"
        return xml_path, bin_path

    def download_openvino_model(self) -> bool:
        """Download pre-converted OpenVINO model files"""
        xml_path, bin_path = self.get_model_paths()
        
        # Check if both files already exist
        if (xml_path.exists() and xml_path.stat().st_size > 0 and 
            bin_path.exists() and bin_path.stat().st_size > 0):
            print(f"OpenVINO model already available at: {xml_path}")
            return True
        
        print(f"Downloading pre-converted OpenVINO MobileNet SSD model...")
        
        try:
            # Download XML file
            print(f"Downloading XML file...")
            print(f"URL: {self.model_xml_url}")
            response = requests.get(self.model_xml_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(xml_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rXML download progress: {progress:.1f}%", end='', flush=True)
            
            print(f"\nXML file downloaded: {xml_path} ({downloaded} bytes)")
            
            # Download BIN file
            print(f"Downloading BIN file...")
            print(f"URL: {self.model_bin_url}")
            response = requests.get(self.model_bin_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(bin_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rBIN download progress: {progress:.1f}%", end='', flush=True)
            
            print(f"\nBIN file downloaded: {bin_path} ({downloaded} bytes)")
            
            # Verify files were downloaded properly
            if xml_path.stat().st_size == 0 or bin_path.stat().st_size == 0:
                raise Exception("One or more downloaded files are empty")
                
            print("OpenVINO model downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"\nError downloading OpenVINO model: {e}")
            if xml_path.exists():
                xml_path.unlink()
            if bin_path.exists():
                bin_path.unlink()
            return False

    def load_openvino_model(self) -> bool:
        """Load pre-converted OpenVINO model"""
        xml_path, bin_path = self.get_model_paths()
        
        if not xml_path.exists() or not bin_path.exists():
            print(f"OpenVINO model files not found: {xml_path}, {bin_path}")
            return False
            
        try:
            print(f"Loading OpenVINO model from {xml_path}")
            self.ov_model = self.core.read_model(str(xml_path))
            
            # Check if model has dynamic input shapes and reshape if needed
            input_layer = self.ov_model.input()
            if input_layer.partial_shape.is_dynamic:
                print("Model has dynamic input shape, reshaping to fixed dimensions...")
                # Reshape to fixed input size: [1, height, width, 3]
                new_input_shape = [1, self.input_size[1], self.input_size[0], 3]
                
                # Create reshape dictionary with input shape
                reshape_dict = {input_layer: new_input_shape}
                
                self.ov_model.reshape(reshape_dict)
                print(f"Model input reshaped to: {new_input_shape}")
            
            print("OpenVINO model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading OpenVINO model: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
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

    def load_or_download_model(self, device: str = None):
        """Load existing OpenVINO model or download pre-converted model"""
        xml_path, bin_path = self.get_model_paths()
        
        # Auto-detect best device if not specified
        if device is None:
            device = self.get_best_device()
        
        # Try to load existing OpenVINO model first
        if xml_path.exists() and bin_path.exists():
            if not self.load_openvino_model():
                print("Failed to load existing model, will download new one...")
                self.ov_model = None
        
        # If no existing model, download pre-converted OpenVINO model
        if self.ov_model is None:
            print("OpenVINO model not found. Downloading pre-converted model...")
            
            # Download OpenVINO model
            if not self.download_openvino_model():
                print("Failed to download OpenVINO model")
                return False
            
            # Load the downloaded model
            if not self.load_openvino_model():
                print("Failed to load downloaded OpenVINO model")
                return False
        
        # Compile model for inference
        try:
            print(f"Compiling model for {device} device...")
            self.compiled_model = self.core.compile_model(self.ov_model, device)
            
            # Debug: Print model input/output info
            if self.debug:
                print("Model input information:")
                for input_layer in self.ov_model.inputs:
                    try:
                        shape_str = str(input_layer.shape)
                    except:
                        shape_str = str(input_layer.partial_shape)
                    print(f"  Input: {input_layer.any_name}, shape: {shape_str}, type: {input_layer.element_type}")
                print("Model output information:")
                for output_layer in self.ov_model.outputs:
                    try:
                        shape_str = str(output_layer.shape)
                    except:
                        shape_str = str(output_layer.partial_shape)
                    print(f"  Output: {output_layer.any_name}, shape: {shape_str}, type: {output_layer.element_type}")
            
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
        """Preprocess image for SSD MobileNet inference"""
        # Resize to model input size
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Model expects uint8 input, so keep as uint8
        # Add batch dimension
        batched = np.expand_dims(rgb, axis=0)
        
        return batched

    def postprocess_detections(self, outputs: List[np.ndarray], original_shape: Tuple[int, int]) -> List[Dict]:
        """Post-process SSDLite MobileDet outputs to get final detections"""
        detections = []
        
        if self.debug:
            print(f"Number of outputs: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"Output {i}: shape {output.shape}, dtype {output.dtype}")
        
        try:
            # For model without post-processing, we typically get raw outputs:
            # - Bounding box coordinates
            # - Class predictions/logits
            # We need to implement our own NMS and post-processing
            
            if len(outputs) == 2:
                # Common format: [boxes, class_predictions]
                boxes = outputs[0]  # Shape: (1, num_anchors, 4)
                class_predictions = outputs[1]  # Shape: (1, num_anchors, num_classes)
                
                if len(boxes.shape) == 3:
                    boxes = boxes[0]  # Remove batch dimension
                if len(class_predictions.shape) == 3:
                    class_predictions = class_predictions[0]  # Remove batch dimension
                
                # Apply softmax to get class probabilities
                class_probs = self._softmax(class_predictions)
                
                # Get the class with highest probability for each detection
                class_ids = np.argmax(class_probs, axis=1)
                confidences = np.max(class_probs, axis=1)
                
            elif len(outputs) == 4:
                # Format with separate outputs (already post-processed)
                boxes = outputs[0][0]  # Shape: (10, 4)
                classes = outputs[1][0]  # Shape: (10,)
                scores = outputs[2][0]  # Shape: (10,)
                num_detections = int(outputs[3][0])  # Scalar
                
                if self.debug:
                    print(f"Number of detections: {num_detections}")
                    print(f"Score range: {np.min(scores):.3f} - {np.max(scores):.3f}")
                
                # Use only the valid detections
                num_detections = min(num_detections, len(boxes))
                
                img_h, img_w = original_shape
                
                for i in range(num_detections):
                    score = scores[i]
                    class_id = int(classes[i])
                    
                    if self.debug:
                        print(f"  Detection {i}: class_id={class_id}, score={score:.3f}")
                    
                    # Filter by confidence threshold and wanted classes
                    if score < self.confidence_threshold:
                        if self.debug:
                            print(f"    Skipped: score {score:.3f} < threshold {self.confidence_threshold}")
                        continue
                        
                    if class_id not in self.wanted_class_ids:
                        if self.debug:
                            print(f"    Skipped: class_id {class_id} not in wanted classes {self.wanted_class_ids}")
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
                
                return detections
            
            else:
                print(f"Unexpected number of outputs: {len(outputs)}")
                return detections
            
            # For raw outputs, we need to implement NMS and filtering
            img_h, img_w = original_shape
            
            # Apply NMS
            valid_detections = []
            for i in range(len(boxes)):
                confidence = confidences[i]
                class_id = class_ids[i]
                
                # Filter by confidence threshold and wanted classes
                if confidence < self.confidence_threshold:
                    continue
                    
                if class_id not in self.wanted_class_ids:
                    continue
                
                # Convert normalized coordinates to pixel coordinates
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
                
                valid_detections.append({
                    'class_id': class_id,
                    'class_name': self.class_names[class_id],
                    'confidence': float(confidence),
                    'bbox': [x, y, w, h]
                })
            
            # Apply NMS to remove overlapping detections
            if valid_detections:
                detections = self._apply_nms(valid_detections)
            
        except Exception as e:
            print(f"Error processing detections: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return detections
        
        return detections
    
    def _softmax(self, x):
        """Apply softmax to convert logits to probabilities"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return detections
        
        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes = []
        confidences = []
        class_ids = []
        
        for det in detections:
            x, y, w, h = det['bbox']
            boxes.append([x, y, w, h])
            confidences.append(det['confidence'])
            class_ids.append(det['class_id'])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Return filtered detections
        filtered_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                filtered_detections.append(detections[i])
        
        return filtered_detections

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
    parser.add_argument('--image', '-i', type=str, help='Test with a single image instead of video')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SSDLite MobileDet OpenVINO Object Detector")
    print("=" * 60)
    
    detector = SSDLiteMobileDetOpenVINODetector(debug=args.debug)
    detector.confidence_threshold = args.threshold
    
    # Load model
    if not detector.load_or_download_model():
        print("❌ Failed to load model.")
        return
    
    # Test with single image if specified
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"Testing with image: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        # Run detection
        start_time = time.time()
        detections = detector.detect(image)
        inference_time = time.time() - start_time
        
        print(f"✅ Detection completed in {inference_time:.3f}s")
        print(f"Found {len(detections)} objects:")
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")
        
        # Draw results
        result_image = detector.draw_detections(image, detections)
        
        # Save result
        output_path = image_path.parent / f"{image_path.stem}_detected{image_path.suffix}"
        cv2.imwrite(str(output_path), result_image)
        print(f"Result saved to: {output_path}")
        
        # Display result
        cv2.imshow('Detection Result', result_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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