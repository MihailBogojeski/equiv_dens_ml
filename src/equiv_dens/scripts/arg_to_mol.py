# %%
import glob
import os
import sys

src_file = sys.argv[1]
dst_mol = sys.argv[2]

print(f"Source file: {src_file}")
print(f"Destination molecule: {dst_mol}")

args_dir = '/Users/nicokestel/Desktop/equiv_dens_ml/args'
os.chdir(args_dir)

split_sizes = {
    'water': {
        'train': 500,
        'valid': 500,
        'test': 3999
    },
    'ethanol': {
        'train': 25000,
        'valid': 500,
        'test': 4500
    },
    'mda-enol': {
        'train': 25000,
        'valid': 500,
        'test': 1478
    },
    'uracil': {
        'train': 25000,
        'valid': 500,
        'test': 4500
    }
}

if "h2o" in src_file:
    src_mol = "water"
elif "ethanol" in src_file:
    src_mol = "ethanol"
elif "mda-enol" in src_file:
    src_mol = "mda-enol"
elif "uracil" in src_file:
    src_mol = "uracil"
else:
    raise ValueError(f"Unknown molecule in source file: {src_file}")

with open(src_file, 'r') as f:
    lines = f.readlines()
    new_lines = []

    for line in lines:

        print(f"Processing line: {line.strip()}")

        if line.startswith("--args_file_name"):
            new_lines.append(line.replace(src_mol, dst_mol))

        elif line.startswith("--np_dataset") or line.startswith("--dens_dataset"):
            new_lines.append(line.replace(src_mol, dst_mol))

        elif line.startswith("--num_train"):
            new_lines.append(f"--num_train={split_sizes[dst_mol]['train']}\n")
        elif line.startswith("--num_valid"):
            new_lines.append(f"--num_valid={split_sizes[dst_mol]['valid']}\n")
        elif line.startswith("--num_test"):
            new_lines.append(f"--num_test={split_sizes[dst_mol]['test']}\n")

        else:
            new_lines.append(line)

        # print(f"Updated line: {line.strip()}")
        

    new_file = src_file
    if src_mol == "water":
        new_file = new_file.replace("h2o", "water")
    new_file = new_file.replace(src_mol, dst_mol)
    with open(new_file, 'w') as new_f:
        print(f"Writing to new file: {new_file}")
        new_f.writelines(new_lines)

