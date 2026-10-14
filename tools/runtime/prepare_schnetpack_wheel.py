"""Normalize SchNetPack 2.1.1 metadata for a validated modern Torch runtime.

No package source is changed. The local version identifies the relaxed SymPy
requirement, and all wheel RECORD hashes are regenerated. Fail closed if the
expected upstream version or dependency cap changes.
"""

import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path
import re
import zipfile


def prepare(source, destination):
    source = Path(source)
    if source.name != 'schnetpack-2.1.1-py3-none-any.whl':
        raise ValueError('Expected standard upstream wheel filename')
    with zipfile.ZipFile(source) as wheel:
        files = {entry.filename: (entry, wheel.read(entry))
                 for entry in wheel.infolist() if not entry.is_dir()}
    metadata_paths = [name for name in files if name.endswith('.dist-info/METADATA')]
    if len(metadata_paths) != 1:
        raise ValueError('Expected exactly one wheel METADATA file')
    metadata_path = metadata_paths[0]
    info, payload = files[metadata_path]
    metadata = payload.decode('utf-8')
    if not re.search(r'^Name: schnetpack\s*$', metadata, re.M | re.I):
        raise ValueError('Expected SchNetPack wheel')
    if not re.search(r'^Version: 2\.1\.1\s*$', metadata, re.M):
        raise ValueError('Expected upstream version 2.1.1')
    metadata, count = re.subn(r'^Requires-Dist: sympy\s*<=\s*1\.12[ \t]*$',
                             'Requires-Dist: sympy>=1.13.3', metadata, flags=re.M)
    if count != 1:
        raise ValueError('Expected exactly the upstream SymPy <=1.12 cap')
    metadata = re.sub(r'^Version: 2\.1\.1[ \t]*$',
                      'Version: 2.1.1+densnet1', metadata, flags=re.M)
    old_directory = metadata_path.rsplit('/', 1)[0]
    new_directory = old_directory.replace('2.1.1', '2.1.1+densnet1')
    rewritten = {}
    for name, (entry, data) in files.items():
        if name == metadata_path:
            data = metadata.encode('utf-8')
        name = name.replace(old_directory + '/', new_directory + '/', 1)
        rewritten[name] = (entry, data)
    record_path = new_directory + '/RECORD'
    if record_path not in rewritten:
        raise ValueError('Expected wheel RECORD')
    rows = []
    for name, (_, data) in sorted(rewritten.items()):
        if name == record_path:
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        rows.append((name, 'sha256=' + digest, str(len(data))))
    rows.append((record_path, '', ''))
    record = io.StringIO(newline='')
    csv.writer(record, lineterminator='\n').writerows(rows)
    rewritten[record_path] = (rewritten[record_path][0], record.getvalue().encode())
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name.replace('-2.1.1-', '-2.1.1+densnet1-', 1)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as wheel:
        for name, (entry, data) in sorted(rewritten.items()):
            new_entry = zipfile.ZipInfo(name, entry.date_time)
            new_entry.external_attr = entry.external_attr
            new_entry.compress_type = zipfile.ZIP_DEFLATED
            wheel.writestr(new_entry, data)
    print(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    prepare(args.source, args.destination)
