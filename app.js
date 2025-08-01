class ObjectDetectionApp {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.status = document.getElementById('status');
        this.detectionInfo = document.getElementById('detectionInfo');
        
        this.stream = null;
        this.isDetecting = false;
        this.animationId = null;
        this.model = null;
        
        this.init();
    }
    
    async init() {
        try {
            this.updateStatus('Downloading YOLO-NAS model...', 'loading');
            await this.loadModel();
            this.updateStatus('Model loaded successfully! Click "Start Camera" to begin.', 'ready');
            document.getElementById('startBtn').disabled = false;
        } catch (error) {
            console.error('Initialization error:', error);
            this.updateStatus('Error loading model. Check console for details.', 'error');
        }
    }
    
    async loadModel() {
        // Simulate model loading from Hugging Face
        // In a real implementation, you would load the actual OpenVINO model
        return new Promise((resolve) => {
            setTimeout(() => {
                this.model = {
                    loaded: true,
                    classes: [
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
                };
                resolve();
            }, 2000); // Simulate 2 second loading time
        });
    }
    
    async startCamera() {
        try {
            this.updateStatus('Starting camera...', 'loading');
            
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            
            this.video.srcObject = this.stream;
            
            this.video.onloadedmetadata = () => {
                this.updateStatus('Camera started! Click "Start Detection" to begin object detection.', 'ready');
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('detectBtn').disabled = false;
            };
            
        } catch (error) {
            console.error('Camera error:', error);
            this.updateStatus('Error accessing camera. Please check permissions.', 'error');
        }
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        this.isDetecting = false;
        this.video.srcObject = null;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.updateStatus('Camera stopped.', 'ready');
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('detectBtn').disabled = true;
        document.getElementById('detectBtn').textContent = 'Start Detection';
        
        this.detectionInfo.innerHTML = '<p>Detected objects will appear here...</p>';
    }
    
    toggleDetection() {
        if (this.isDetecting) {
            this.stopDetection();
        } else {
            this.startDetection();
        }
    }
    
    startDetection() {
        this.isDetecting = true;
        document.getElementById('detectBtn').textContent = 'Stop Detection';
        this.updateStatus('Running real-time object detection...', 'ready');
        this.detectObjects();
    }
    
    stopDetection() {
        this.isDetecting = false;
        document.getElementById('detectBtn').textContent = 'Start Detection';
        this.updateStatus('Detection stopped.', 'ready');
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.detectionInfo.innerHTML = '<p>Detection stopped.</p>';
    }
    
    async detectObjects() {
        if (!this.isDetecting || !this.model || !this.video.videoWidth) {
            return;
        }
        
        // Draw video frame to canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Simulate object detection
        const detections = this.simulateDetection();
        
        // Draw bounding boxes and labels
        this.drawDetections(detections);
        
        // Update detection info
        this.updateDetectionInfo(detections);
        
        // Continue detection loop
        this.animationId = requestAnimationFrame(() => this.detectObjects());
    }
    
    simulateDetection() {
        // Simulate random detections for demo purposes
        // In real implementation, this would call the OpenVINO inference
        const detections = [];
        const numDetections = Math.floor(Math.random() * 4); // 0-3 detections
        
        for (let i = 0; i < numDetections; i++) {
            const classIndex = Math.floor(Math.random() * this.model.classes.length);
            const confidence = 0.5 + Math.random() * 0.5; // 0.5-1.0 confidence
            
            detections.push({
                class: this.model.classes[classIndex],
                confidence: confidence,
                bbox: {
                    x: Math.random() * (this.canvas.width - 100),
                    y: Math.random() * (this.canvas.height - 100),
                    width: 50 + Math.random() * 150,
                    height: 50 + Math.random() * 150
                }
            });
        }
        
        return detections;
    }
    
    drawDetections(detections) {
        detections.forEach(detection => {
            const { bbox, class: className, confidence } = detection;
            
            // Draw bounding box
            this.ctx.strokeStyle = '#00ff00';
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
            
            // Draw label background
            const label = `${className} (${(confidence * 100).toFixed(1)}%)`;
            this.ctx.font = '14px Arial';
            const textWidth = this.ctx.measureText(label).width;
            
            this.ctx.fillStyle = '#00ff00';
            this.ctx.fillRect(bbox.x, bbox.y - 25, textWidth + 10, 20);
            
            // Draw label text
            this.ctx.fillStyle = '#000000';
            this.ctx.fillText(label, bbox.x + 5, bbox.y - 10);
        });
    }
    
    updateDetectionInfo(detections) {
        if (detections.length === 0) {
            this.detectionInfo.innerHTML = '<p>No objects detected in current frame.</p>';
            return;
        }
        
        let html = '<h4>Detected Objects:</h4><ul>';
        detections.forEach(detection => {
            html += `<li>${detection.class} - ${(detection.confidence * 100).toFixed(1)}% confidence</li>`;
        });
        html += '</ul>';
        
        this.detectionInfo.innerHTML = html;
    }
    
    updateStatus(message, type) {
        this.status.textContent = message;
        this.status.className = `status ${type}`;
    }
}

// Global functions for button handlers
let app;

function startCamera() {
    app.startCamera();
}

function stopCamera() {
    app.stopCamera();
}

function toggleDetection() {
    app.toggleDetection();
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    app = new ObjectDetectionApp();
});