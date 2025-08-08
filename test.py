from openvino.runtime import Core

core = Core()
model = core.read_model(model=".\models\ssd_mobilenet_v2_coco.xml")
inputs = model.inputs
for input in inputs:
    print(f"Input: {input.get_any_name()}, Shape: {input.get_partial_shape()}")