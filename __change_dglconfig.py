import os
import json

backend_name = 'pytorch'  # Change to 'pytorch' for PyTorch backend

# to avoid import error at dgl/backend/__init__.py
default_dir = None
if "DGLDEFAULTDIR" in os.environ:
    default_dir = os.getenv("DGLDEFAULTDIR")
else:
    default_dir = os.path.join(os.path.expanduser("~"), ".dgl")
config_path = os.path.join(default_dir, "config.json")

# Read the current configuration
with open(config_path, 'r') as input_file:
    _config = json.load(input_file)
    _config['backend'] = backend_name  # Update the backend

# Write the updated configuration back to the file
with open(config_path, 'w') as output_file:
    json.dump(_config, output_file, indent=4)
