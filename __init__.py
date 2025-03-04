import importlib.util
import os
import sys
import folder_paths

current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

def load_modules_from_directory(directory):
    if not os.path.exists(directory):
        return

    for file in os.listdir(directory):
        if not file.endswith(".py"):
            continue
            
        module_name = os.path.basename(file)[:-3]
        if module_name == os.path.basename(__file__)[:-3]:
            continue

        file_path = os.path.join(directory, file)
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                continue
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                
        except Exception as e:
            print(f"Error loading module {module_name}: {e}")

def load_javascript(web_directory):
    return []

load_modules_from_directory(current_dir)
load_modules_from_directory(os.path.join(current_dir, "py"))
# load_modules_from_directory(os.path.join(current_dir, "test_nodes"))

NODE_CLASS_MAPPINGS = dict(sorted(
    NODE_CLASS_MAPPINGS.items(),
    key=lambda x: NODE_DISPLAY_NAME_MAPPINGS.get(x[0], x[0])
))
NODE_DISPLAY_NAME_MAPPINGS = dict(sorted(
    NODE_DISPLAY_NAME_MAPPINGS.items(),
    key=lambda x: x[1]
))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "load_javascript"]