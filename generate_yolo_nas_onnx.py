#!/usr/bin/env python3
"""
Script to generate YOLO-NAS ONNX models using super-gradients library.
Run this to create real YOLO-NAS ONNX models for OpenVINO inference.
"""

def generate_yolo_nas_onnx():
    try:
        from super_gradients.training import models
        import torch
        from pathlib import Path
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Available models
        model_names = ["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]
        
        for model_name in model_names:
            print(f"Loading {model_name}...")
            
            # Load pre-trained model
            model = models.get(model_name, pretrained_weights="coco")
            model.eval()
            
            # Export to ONNX
            output_path = models_dir / f"{model_name}.onnx"
            print(f"Exporting to {output_path}...")
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 640, 640)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
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
            
            print(f"✓ {model_name} exported successfully to {output_path}")
            
    except ImportError:
        print("super-gradients not installed. Install with:")
        print("pip install super-gradients")
        return False
    except Exception as e:
        print(f"Error generating ONNX models: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Generating YOLO-NAS ONNX models...")
    if generate_yolo_nas_onnx():
        print("All models generated successfully!")
    else:
        print("Failed to generate models. Check the error messages above.")