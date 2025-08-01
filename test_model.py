#!/usr/bin/env python3
"""
Test script to check the SSD MobileNet model structure
"""

import openvino as ov
from pathlib import Path

def test_model():
    core = ov.Core()
    model_path = Path("models/ssd_mobilenet_v1.xml")
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    try:
        print(f"Loading model: {model_path}")
        model = core.read_model(str(model_path))
        
        print("\nInput information:")
        for i, input_layer in enumerate(model.inputs):
            print(f"  Input {i}: {input_layer.any_name}")
            try:
                print(f"    Shape: {input_layer.shape}")
            except:
                print(f"    Shape: Dynamic")
            print(f"    Partial shape: {input_layer.partial_shape}")
            print(f"    Element type: {input_layer.element_type}")
            print(f"    Is dynamic: {input_layer.partial_shape.is_dynamic}")
        
        print("\nOutput information:")
        for i, output_layer in enumerate(model.outputs):
            print(f"  Output {i}: {output_layer.any_name}")
            try:
                print(f"    Shape: {output_layer.shape}")
            except:
                print(f"    Shape: Dynamic")
            print(f"    Partial shape: {output_layer.partial_shape}")
            print(f"    Element type: {output_layer.element_type}")
            print(f"    Is dynamic: {output_layer.partial_shape.is_dynamic}")
        
        # Try to reshape the model
        print("\nTrying to reshape model...")
        input_layer = model.input()
        new_shape = [1, 300, 300, 3]
        print(f"Reshaping input to: {new_shape}")
        
        model.reshape({input_layer: new_shape})
        
        print("\nAfter reshape - Input information:")
        for i, input_layer in enumerate(model.inputs):
            print(f"  Input {i}: {input_layer.any_name}")
            try:
                print(f"    Shape: {input_layer.shape}")
            except:
                print(f"    Shape: Dynamic")
            print(f"    Is dynamic: {input_layer.partial_shape.is_dynamic}")
        
        print("\nAfter reshape - Output information:")
        for i, output_layer in enumerate(model.outputs):
            print(f"  Output {i}: {output_layer.any_name}")
            try:
                print(f"    Shape: {output_layer.shape}")
            except:
                print(f"    Shape: Dynamic")
            print(f"    Is dynamic: {output_layer.partial_shape.is_dynamic}")
        
        # Try to compile
        print("\nTrying to compile model...")
        compiled_model = core.compile_model(model, "CPU")
        print("✅ Model compiled successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()