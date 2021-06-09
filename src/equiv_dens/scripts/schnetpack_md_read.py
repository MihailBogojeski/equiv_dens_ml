from schnetpack.md.utils import HDF5Loader
import sys

log_file = sys.argv[1]

data = HDF5Loader(log_file)

print(data.properties.keys())
print(data.properties['_positions'])
print(data.properties['_positions'].shape)
