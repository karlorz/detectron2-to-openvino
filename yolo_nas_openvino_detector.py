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
import requests
import os
import sys
import argparse
from typing import List, Dict, Tuple
import warnings
import platform

class YOLONASOpenVINODetector:
    def __init__(self, model_size: str = "yolo_nas_m", debug: bool = False):
        """
        Initialize YOLO-NAS OpenVINO detector
        
        Args:
            model_size: One of 'yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'
            debug: Enable debug output
        """
        self.model_size = model_size
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.debug = debug
        
        self.confidence_threshold = 0.3  # Reasonable threshold for real use
        self.nms_threshold = 0.4
        self.input_size = (640, 640)
        
        self.ov_model = None
        self.compiled_model = None
        self.core = ov.Core()
        
        # Full COCO class names (80 classes) - model outputs these IDs
        self.all_class_names = [
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
        
        # Classes you want to detect (filter)
        self.wanted_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck'
        ]
        
        # Create mapping from class ID to class name and filter set
        self.wanted_class_ids = set()
        for class_name in self.wanted_classes:
            if class_name in self.all_class_names:
                class_id = self.all_class_names.index(class_name)
                self.wanted_class_ids.add(class_id)
        
        if debug:
            print(f"Filtering to detect only: {self.wanted_classes}")
            print(f"Class IDs to detect: {sorted(self.wanted_class_ids)}")
        
        # Generate colors for each class (use full set for consistent colors)
        np.random.seed(42)  # For consistent colors
        self.colors = np.random.uniform(0, 255, size=(len(self.all_class_names), 3))
        
        # Model URLs for downloading YOLO-NAS ONNX models
        self.model_urls = {
            "yolo_nas_s": "https://huggingface.co/hr16/yolo-nas-fp16/resolve/main/yolo_nas_s_fp16.onnx",
            "yolo_nas_m": "https://huggingface.co/hr16/yolo-nas-fp16/resolve/main/yolo_nas_m_fp16.onnx",
            "yolo_nas_l": "https://huggingface.co/hr16/yolo-nas-fp16/resolve/main/yolo_nas_l_fp16.onnx"
        }
        
        # Alternative: Generate models using super-gradients if URLs fail
        self.use_super_gradients_fallback = True

    def generate_onnx_with_super_gradients(self) -> bool:
        """Generate ONNX model using super-gradients library"""
        onnx_path = self.model_dir / f"{self.model_size}.onnx"
        
        try:
            print(f"Attempting to generate {self.model_size} model using super-gradients...")
            from super_gradients.training import models
            
            # Load pre-trained model
            model = models.get(self.model_size, pretrained_weights="coco")
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 640, 640)
            
            print(f"Exporting model to ONNX format...")
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"Model generated successfully: {onnx_path}")
            return True
            
        except ImportError:
            print("super-gradients not installed. Install with:")
            print("uv add super-gradients")
            return False
        except Exception as e:
            print(f"Error generating model with super-gradients: {e}")
            if onnx_path.exists():
                onnx_path.unlink()
            return False

    def download_onnx_model(self) -> bool:
        """Download ONNX model or generate it locally if download fails"""
        onnx_path = self.model_dir / f"{self.model_size}.onnx"
        
        if onnx_path.exists() and onnx_path.stat().st_size > 0:
            print(f"ONNX model {onnx_path} already exists locally")
            return True
        
        # Try downloading first
        if self.model_size in self.model_urls:
            url = self.model_urls[self.model_size]
            print(f"Downloading ONNX model from: {url}")
            
            try:
                # Add headers to avoid 401 errors
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, stream=True, headers=headers, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(onnx_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\rDownload progress: {progress:.1f}%", end='', flush=True)
                
                print(f"\nModel downloaded successfully to {onnx_path}")
                return True
                
            except Exception as e:
                print(f"Download failed: {e}")
                if onnx_path.exists():
                    onnx_path.unlink()  # Remove incomplete file
        
        # Fallback to super-gradients generation
        if self.use_super_gradients_fallback:
            print("Falling back to local model generation...")
            return self.generate_onnx_with_super_gradients()
        
        return False

    def convert_onnx_to_openvino(self) -> bool:
        """Convert ONNX model to OpenVINO format"""
        onnx_path = self.model_dir / f"{self.model_size}.onnx"
        xml_path = self.model_dir / f"{self.model_size}.xml"
        
        if not onnx_path.exists():
            print(f"ONNX model not found: {onnx_path}")
            return False
            
        try:
            print(f"Converting ONNX model to OpenVINO format...")
            
            # Load and convert ONNX model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ov_model = ov.convert_model(str(onnx_path))
            
            # Save OpenVINO model
            print(f"Saving OpenVINO model to {xml_path}")
            ov.save_model(ov_model, str(xml_path))
            
            self.ov_model = ov_model
            print("Model conversion completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error converting ONNX to OpenVINO: {e}")
            return False

    def get_best_device(self) -> str:
        """Detect the best available device (GPU preferred on Windows)"""
        available_devices = self.core.available_devices
        print(f"Available devices: {available_devices}")
        
        # On Windows, prefer GPU if available
        if platform.system() == "Windows":
            if "GPU.0" in available_devices or "GPU" in available_devices:
                print("Using GPU device for inference")
                return "GPU"
        
        # Check for other GPU devices
        for device in available_devices:
            if "GPU" in device:
                print(f"Using {device} device for inference")
                return device
                
        print("Using CPU device for inference")
        return "CPU"

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

    def load_or_convert_model(self, device: str = None):
        """Load existing OpenVINO model or download and convert from ONNX"""
        model_xml_path = self.model_dir / f"{self.model_size}.xml"
        
        # Auto-detect best device if not specified
        if device is None:
            device = self.get_best_device()
        
        # Try to load existing OpenVINO model first
        if model_xml_path.exists() and (self.model_dir / f"{self.model_size}.bin").exists():
            try:
                print(f"Loading existing OpenVINO model from {model_xml_path}")
                self.ov_model = self.core.read_model(str(model_xml_path))
            except Exception as e:
                print(f"Error loading existing model: {e}")
                print("Will download and convert a new model...")
                self.ov_model = None
        
        # If no existing model, download and convert
        if self.ov_model is None:
            print("OpenVINO model not found. Downloading ONNX model...")
            
            # Download ONNX model
            if not self.download_onnx_model():
                print("Failed to download ONNX model")
                return False
            
            # Convert ONNX to OpenVINO
            if not self.convert_onnx_to_openvino():
                print("Failed to convert ONNX model to OpenVINO")
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
        """Preprocess image for YOLO-NAS inference"""
        # Simple resize without letterbox
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB (OpenCV uses BGR, models expect RGB)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and apply ImageNet normalization (works best for this model)
        normalized = rgb.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Convert HWC to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched

    def postprocess_detections(self, outputs: List[np.ndarray], original_shape: Tuple[int, int]) -> List[Dict]:
        """Post-process YOLO-NAS outputs to get final detections"""
        detections = []
        
        if not outputs or len(outputs) < 4:
            print(f"Expected 4 outputs, got {len(outputs)}")
            return detections
        
        # YOLO-NAS specific output format:
        # Output 0: num_predictions (1,1) - number of valid predictions
        # Output 1: pred_boxes (1,1000,4) - bounding boxes  
        # Output 2: pred_scores (1,1000) - confidence scores
        # Output 3: pred_classes (1,1000) - class IDs
        
        try:
            num_predictions = outputs[0][0, 0]  # Extract scalar value
            boxes = outputs[1][0]  # Remove batch dimension: (1000, 4)
            scores = outputs[2][0]  # Remove batch dimension: (1000,)
            class_ids = outputs[3][0]  # Remove batch dimension: (1000,)
            
            if self.debug:
                print(f"Number of predictions: {num_predictions}")
                print(f"Boxes shape: {boxes.shape}")
                print(f"Scores shape: {scores.shape}")
                print(f"Classes shape: {class_ids.shape}")
                print(f"Score range: {np.min(scores):.3f} - {np.max(scores):.3f}")
            
            # Use only the valid predictions (first num_predictions)
            if num_predictions > 0:
                num_pred = min(int(num_predictions), len(boxes))
                boxes = boxes[:num_pred]
                scores = scores[:num_pred]
                class_ids = class_ids[:num_pred]
                
                if self.debug:
                    print(f"Using first {num_pred} predictions")
                    print(f"Filtered score range: {np.min(scores):.3f} - {np.max(scores):.3f}")
            else:
                if self.debug:
                    print("No valid predictions from model")
                return detections
                
        except Exception as e:
            print(f"Error extracting outputs: {e}")
            return detections
        
        # Filter by confidence threshold
        valid_mask = scores >= self.confidence_threshold
        valid_count = np.sum(valid_mask)
        
        if self.debug:
            print(f"Valid detections: {valid_count} out of {len(scores)} (threshold: {self.confidence_threshold})")
        
        if valid_count == 0:
            return detections
        
        # Filter predictions by confidence and wanted classes
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]
        
        # Further filter by wanted classes only
        class_filter_mask = np.array([int(class_id) in self.wanted_class_ids for class_id in class_ids])
        
        if not np.any(class_filter_mask):
            return detections
            
        boxes = boxes[class_filter_mask]
        scores = scores[class_filter_mask]
        class_ids = class_ids[class_filter_mask]
        
        # Scale boxes back to original image size (simple scaling without letterbox)
        img_h, img_w = original_shape
        target_h, target_w = self.input_size
        
        # Simple scaling factors
        scale_x = img_w / target_w
        scale_y = img_h / target_h
        
        # Scale coordinates back to original image size
        boxes[:, [0, 2]] *= scale_x  # x coordinates
        boxes[:, [1, 3]] *= scale_y  # y coordinates
        
        # Convert to [x, y, w, h] format for NMS
        boxes_xywh = np.zeros_like(boxes)
        boxes_xywh[:, 0] = boxes[:, 0]  # x1 -> x
        boxes_xywh[:, 1] = boxes[:, 1]  # y1 -> y
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1
        
        # Apply NMS
        try:
            indices = cv2.dnn.NMSBoxes(
                boxes_xywh.tolist(),
                scores.tolist(),
                self.confidence_threshold,
                self.nms_threshold
            )
            
            if len(indices) > 0:
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                elif isinstance(indices, tuple):
                    indices = [indices]
                
                for i in indices:
                    if isinstance(i, (list, tuple, np.ndarray)):
                        i = i[0] if hasattr(i, '__len__') else i
                    
                    x, y, w, h = boxes_xywh[i]
                    confidence = scores[i]
                    class_id = class_ids[i]
                    
                    # Ensure coordinates are within bounds
                    x = max(0, min(int(x), img_w - 1))
                    y = max(0, min(int(y), img_h - 1))
                    w = max(1, min(int(w), img_w - x))
                    h = max(1, min(int(h), img_h - y))
                    
                    # Ensure class_id is within bounds and is a wanted class
                    if 0 <= class_id < len(self.all_class_names) and class_id in self.wanted_class_ids:
                        detections.append({
                            'class_id': int(class_id),
                            'class_name': self.all_class_names[int(class_id)],
                            'confidence': float(confidence),
                            'bbox': [x, y, w, h]
                        })
        
        except Exception as e:
            print(f"Error in NMS: {e}")
            # Fallback: return top detections without NMS
            top_indices = np.argsort(scores)[-10:]  # Top 10
            for i in top_indices:
                x, y, w, h = boxes_xywh[i]
                confidence = scores[i]
                class_id = class_ids[i]
                
                x = max(0, min(int(x), img_w - 1))
                y = max(0, min(int(y), img_h - 1))
                w = max(1, min(int(w), img_w - x))
                h = max(1, min(int(h), img_h - y))
                
                if 0 <= class_id < len(self.all_class_names) and class_id in self.wanted_class_ids:
                    detections.append({
                        'class_id': int(class_id),
                        'class_name': self.all_class_names[int(class_id)],
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
        
        # Convert outputs to list in correct order
        output_arrays = []
        output_keys = list(outputs.keys())
        
        # Sort outputs by name to ensure consistent order
        # Expected order: num_predictions, pred_boxes, pred_scores, pred_classes
        sorted_keys = sorted(output_keys)
        for key in sorted_keys:
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLO-NAS OpenVINO Object Detector')
    parser.add_argument('--model', '-m', choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'], 
                       default='yolo_nas_s', help='Model size (default: yolo_nas_m)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--source', '-s', type=str, default='0', 
                       help='Video source: 0 for webcam, URL for stream, path for video file (default: 0)')
    parser.add_argument('--skip-frames', type=int, default=2, 
                       help='Skip every N frames for streams (default: 2, higher = better FPS)')
    parser.add_argument('--buffer-size', type=int, default=1, 
                       help='Video capture buffer size (default: 1, lower = less delay)')
    
    # Check for environment variables
    debug_env = os.getenv('YOLO_DEBUG', '').lower() in ('1', 'true', 'on')
    threshold_env = float(os.getenv('YOLO_THRESHOLD', '0.5'))
    
    args = parser.parse_args()
    debug = args.debug or debug_env
    
    # Use command line threshold, or environment variable, or default
    threshold = args.threshold if args.threshold != 0.5 else threshold_env
    
    print("=" * 60)
    print("YOLO-NAS OpenVINO Object Detector")
    print("=" * 60)
    
    if debug:
        print(f"Debug mode: ON")
    
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {threshold}")
    print(f"Video source: {args.source}")
    print(f"Initializing detector...")
    
    detector = YOLONASOpenVINODetector(args.model, debug=debug)
    detector.confidence_threshold = threshold
    
    # Load model (auto-detects best device)
    if not detector.load_or_convert_model():
        print("❌ Failed to load model.")
        print("")
        print("📋 To fix this, install super-gradients:")
        print("   uv add super-gradients")
        print("")
        print("This will allow the script to generate YOLO-NAS models locally.")
        return
    
    # Test with static image first (if available)
    test_image_path = Path("data/example_image.jpg")
    if test_image_path.exists():
        print(f"Testing detection on {test_image_path}...")
        try:
            test_img = cv2.imread(str(test_image_path))
            if test_img is not None:
                detections = detector.detect(test_img)
                print(f"Test image detection: Found {len(detections)} objects")
                for det in detections:
                    print(f"  - {det['class_name']}: {det['confidence']:.2f}")
        except Exception as e:
            print(f"Error testing with static image: {e}")
    
    # Initialize video source
    def parse_source(source_str):
        """Parse video source string to appropriate format for cv2.VideoCapture"""
        # Try to convert to integer (for webcam index)
        try:
            return int(source_str)
        except ValueError:
            # Return as string (URL or file path)
            return source_str
    
    video_source = parse_source(args.source)
    
    # Determine source type for user feedback
    if isinstance(video_source, int):
        print(f"Initializing webcam (camera {video_source})...")
        source_type = "webcam"
    elif video_source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
        print(f"Connecting to stream: {video_source}")
        source_type = "stream"
    else:
        print(f"Opening video file: {video_source}")
        source_type = "file"
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open {source_type}")
        if source_type == "webcam":
            print("Make sure your webcam is connected and not being used by another application")
        elif source_type == "stream":
            print("Check the stream URL and your internet connection")
        else:
            print("Check if the video file exists and is in a supported format")
        return
    
    # Optimize settings based on source type
    if source_type == "webcam":
        # Webcam optimizations
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    elif source_type == "stream":
        # RTSP/Stream optimizations
        cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffer_size)  # Reduce buffer to minimize delay
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"Stream optimizations: buffer_size={args.buffer_size}, skip_frames={args.skip_frames}")
    
    print(f"✅ {source_type.capitalize()} initialized successfully!")
    print("Starting real-time detection...")
    print("Press 'q' to quit, 'p' to pause/resume")
    
    fps_counter = 0
    start_time = time.time()
    last_fps_time = start_time
    paused = False
    frame_skip_count = 0
    last_detection_time = start_time
    
    # Stream optimization settings
    if source_type == "stream":
        FRAME_SKIP_INTERVAL = args.skip_frames  # Process every Nth frame for streams
        MAX_DETECTION_INTERVAL = 0.3  # Force detection at least every 300ms
    else:
        FRAME_SKIP_INTERVAL = 1  # Process all frames for webcam/files
        MAX_DETECTION_INTERVAL = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if source_type == "stream":
                    # For streams, try to reconnect or skip bad frames
                    if debug:
                        print(f"Stream read failed, attempting to recover...")
                    # Clear buffer and try next frame
                    for _ in range(3):  # Skip a few frames to clear buffer
                        cap.read()
                    continue
                else:
                    if debug:
                        print(f"Warning: Could not read frame from {source_type}")
                    continue
            
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
                continue
            
            if paused:
                cv2.imshow('YOLO-NAS OpenVINO Object Detection', frame)
                continue
            
            # Frame skipping logic for streams
            current_time = time.time()
            frame_skip_count += 1
            should_detect = False
            
            if source_type == "stream":
                # Skip frames for better performance, but ensure minimum detection rate
                if (frame_skip_count % FRAME_SKIP_INTERVAL == 0 or 
                    current_time - last_detection_time > MAX_DETECTION_INTERVAL):
                    should_detect = True
                    last_detection_time = current_time
            else:
                # Process all frames for webcam/files
                should_detect = True
            
            # Run detection only when needed
            if should_detect:
                detections = detector.detect(frame)
            else:
                detections = []  # Use empty list for skipped frames
            
            # Log detections (clean output for performance monitoring)
            if detections:
                current_time = time.time()
                current_fps = fps_counter / (current_time - start_time) if fps_counter > 0 else 0
                
                detection_summary = {}
                for det in detections:
                    class_name = det['class_name']
                    if class_name in detection_summary:
                        detection_summary[class_name] += 1
                    else:
                        detection_summary[class_name] = 1
                
                objects_str = ", ".join([f"{count}x {name}" for name, count in detection_summary.items()])
                print(f"FPS: {current_fps:.1f} | Detected: {objects_str}")
            
            # Draw results
            result_frame = detector.draw_detections(frame, detections)
            
            # Update FPS counter
            fps_counter += 1
            
            # Add info text to frame
            current_fps = fps_counter / (time.time() - start_time) if fps_counter > 0 else 0
            info_text = f"FPS: {current_fps:.1f} | Objects: {len(detections)}"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('YOLO-NAS OpenVINO Object Detection', result_frame)
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    except Exception as e:
        print(f"❌ Error during detection: {e}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Detection stopped and resources cleaned up.")

if __name__ == "__main__":
    main()