# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real-time object detection project that demonstrates converting and running object detection models with OpenVINO. The project supports multiple approaches:

1. **YOLO-NAS with OpenVINO**: Modern approach using YOLO-NAS models converted to ONNX/OpenVINO format
2. **Detectron2 conversion**: Tutorial/notebook showing Detectron2 to OpenVINO conversion
3. **Web demo**: Browser-based demo with simulated detection

## Essential Commands

### Environment Setup (choose one approach)
```bash
# Automated setup (recommended)
python setup.py

# Or use shell script (Unix/macOS)
./run.sh

# Or use batch script (Windows)
run.bat

# Manual setup with uv
uv venv
source .venv/bin/activate  # Unix/macOS
# or .venv\Scripts\activate  # Windows
uv pip install -r requirements.txt
```

### Running the Applications
```bash
# Main Python detector (YOLO-NAS + OpenVINO)
python openvino_detector.py

# Alternative detector implementation
python yolo_nas_openvino_detector.py

# Generate ONNX models from YOLO-NAS
python generate_yolo_nas_onnx.py

# Web demo - open index.html in browser
```

### Development Commands
```bash
# Install dependencies
uv pip install -r requirements.txt

# For Detectron2 specific requirements (if needed)
uv pip install -r requirements_detectron2.txt
```

## Architecture Overview

### Key Components

**Core Detection System:**
- `openvino_detector.py`: Main YOLO-NAS detector with OpenVINO backend
- `yolo_nas_openvino_detector.py`: Alternative YOLO-NAS implementation with native PyTorch conversion
- `YOLONASDetector` and `YOLONASOpenVINODetector` classes: Main detector implementations

**Model Management:**
- Models downloaded from Hugging Face Hub automatically
- Supports YOLO-NAS variants: `yolo_nas_s` (fastest), `yolo_nas_m` (balanced), `yolo_nas_l` (most accurate)
- ONNX conversion and OpenVINO optimization pipeline
- Default model storage in `models/` directory

**Web Interface:**
- `index.html` + `app.js`: Browser-based webcam demo with simulated detection
- Designed for easy integration with Python backend

**Utilities:**
- `generate_yolo_nas_onnx.py`: Script to create real ONNX models using super-gradients library
- `notebook_utils.py`, `pip_helper.py`: Helper modules for Jupyter notebook setup
- `setup.py`: Automated environment setup with cross-platform support

### Detection Pipeline

1. **Model Loading**: Download or load YOLO-NAS models from HuggingFace/local storage
2. **Preprocessing**: Resize, normalize input images to 640x640
3. **Inference**: OpenVINO optimized inference on CPU/GPU
4. **Postprocessing**: NMS filtering, confidence thresholding, coordinate scaling
5. **Visualization**: Bounding box and label rendering with confidence scores

### Configuration Options

Key parameters in detector classes:
- `confidence_threshold`: Default 0.5, minimum confidence for detections
- `nms_threshold`: Default 0.4, non-maximum suppression threshold  
- `input_shape`: Default (640, 640), model input resolution
- Model variants: Configurable via model_size parameter

## Notebook Integration

The `detectron2-to-openvino.ipynb` notebook demonstrates:
- Converting Detectron2 models (Faster R-CNN, Mask R-CNN) to OpenVINO format
- Object detection and instance segmentation examples
- Integration with Detectron2's postprocessing and visualization utilities
- TracingAdapter usage for export-friendly model conversion

## Dependencies

Core dependencies are managed through `pyproject.toml`:
- **openvino**: Intel's inference engine (>=2025.2.0)
- **opencv-python**: Computer vision library
- **torch/torchvision**: PyTorch framework  
- **numpy**: Numerical computing
- **requests**: HTTP library for model downloads
- **pillow**: Image processing

Optional Detectron2 dependencies in separate requirements file.