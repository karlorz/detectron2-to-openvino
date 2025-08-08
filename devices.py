from openvino.runtime import Core

core = Core()
devices = core.available_devices
print("Available devices:", devices)