"""Install the pinned legacy g-xTB interface used by this project."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def install(manifest, destination):
    specification = json.loads(Path(manifest).read_text())
    root = Path(destination)
    names = {'basisq': '.basisq', 'eeq': '.eeq', 'gxtb': 'gxtb',
             'gxtb_parameters': '.gxtb'}
    if {a['name'] for a in specification['assets']} != set(names):
        raise ValueError('Expected the four pinned g-xTB release assets')
    for asset in specification['assets']:
        directory = root / ('bin' if asset['name'] == 'gxtb' else 'parameters')
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / names[asset['name']]
        urllib.request.urlretrieve(asset['url'], target)
        payload = target.read_bytes()
        if len(payload) != asset['size'] or hashlib.sha256(payload).hexdigest() != asset['sha256']:
            target.unlink()
            raise ValueError('g-xTB asset checksum/size mismatch: ' + asset['name'])
        target.chmod(0o755 if asset['name'] == 'gxtb' else 0o644)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('destination')
    args = parser.parse_args()
    install(args.manifest, args.destination)
