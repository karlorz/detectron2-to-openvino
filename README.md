# Real-time Object Detection with OpenVINO and YOLO-NAS

This project demonstrates real-time object detection using OpenVINO with YOLO-NAS models, featuring both web-based and Python implementations.

## Features

- **Web App**: Browser-based webcam capture with simulated object detection
- **Python Backend**: OpenVINO-powered object detection with YOLO-NAS
- **Model Download**: Automatic download from Hugging Face if not available locally
- **Real-time Processing**: Live webcam feed with bounding box visualization
- **Cross-platform**: Works on Windows, macOS, and Linux

## Quick Start

### Automated Setup (Recommended)

**Option 1: Using setup script**
```bash
python setup.py
```

**Option 2: Using shell script (Unix/macOS)**
```bash
./run.sh
```

**Option 3: Using batch script (Windows)**
```bash
run.bat
```

### Manual Setup

**Web Demo**
1. Open `index.html` in your browser
2. Click "Start Camera" to access webcam
3. Click "Start Detection" to begin object detection simulation

**Python Implementation**
1. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh  # Unix/macOS
   # or
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
   ```

2. Create and activate virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # Unix/macOS
   # or
   .venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

4. Run the detector:
   ```bash
   python openvino_detector.py
   ```

5. Press 'q' to quit

## Project Structure

```
├── index.html              # Web interface
├── app.js                  # Frontend JavaScript logic
├── openvino_detector.py    # Python OpenVINO implementation
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Modern Python project configuration
├── setup.py               # Automated setup script
├── run.sh                 # Unix/macOS runner script
├── run.bat                # Windows runner script
└── README.md              # This file
```

## Configuration

### Model Selection
The detector supports different YOLO-NAS variants:
- `yolo_nas_s` (Small - fastest)
- `yolo_nas_m` (Medium - balanced)
- `yolo_nas_l` (Large - most accurate)

### Customization
You can adjust detection parameters in `openvino_detector.py`:
- `confidence_threshold`: Minimum confidence for detections (default: 0.5)
- `nms_threshold`: Non-maximum suppression threshold (default: 0.4)
- `input_shape`: Model input resolution (default: 640x640)

## Dependencies

- **OpenVINO**: Intel's inference engine
- **OpenCV**: Computer vision library
- **NumPy**: Numerical computing
- **Requests**: HTTP library for model downloads
- **Hugging Face Hub**: Model repository access

## Browser Compatibility

The web demo requires:
- Modern browser with WebRTC support
- Camera permissions
- JavaScript enabled

## Performance Tips

1. **GPU Acceleration**: Install OpenVINO GPU plugin for better performance
2. **Model Size**: Use smaller models (yolo_nas_s) for real-time performance
3. **Resolution**: Lower input resolution for faster inference
4. **Batch Processing**: Process multiple frames together when possible

## Troubleshooting

### Common Issues

**Camera Access Denied**
- Check browser permissions
- Ensure HTTPS for production deployment

**Model Download Fails**
- Check internet connection
- Verify Hugging Face model URLs
- Use local model files if available

**Poor Performance**
- Reduce input resolution
- Use smaller model variant
- Enable hardware acceleration

**OpenVINO Installation Issues**
```bash
# Install OpenVINO with uv
uv pip install openvino

# For GPU support (optional)
uv pip install openvino-dev[gpu]
```

## Development

### Adding New Models
1. Update model URLs in `download_model_from_hf()`
2. Adjust preprocessing/postprocessing as needed
3. Update class names if using custom datasets

### Web Integration
The web demo currently simulates detection. To integrate with the Python backend:
1. Set up a Flask/FastAPI server
2. Stream video frames to backend
3. Return detection results as JSON

## License

This project is open source. Check individual model licenses from Hugging Face.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- Intel OpenVINO team
- Deci AI for YOLO-NAS models
- Hugging Face for model hosting