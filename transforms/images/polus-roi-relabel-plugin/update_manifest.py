import json

import toml

with open('pyproject.toml', 'r') as reader:
    version = toml.load(reader)['tool']['poetry']['version']

with open('VERSION', 'w') as writer:
    writer.write(version)

with open('plugin.json', 'r') as reader:
    manifest = json.load(reader)

manifest['version'] = version
manifest['containerId'] = f'polusai/roi-relabel-plugin:{version}'

with open('plugin.json', 'w') as writer:
    json.dump(manifest, writer, indent=2)
