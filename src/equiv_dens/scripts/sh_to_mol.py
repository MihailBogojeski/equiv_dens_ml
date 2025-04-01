# %%
import glob
import os
import sys

src_file = sys.argv[1]
dst_mol = sys.argv[2]

print(f"Source file: {src_file}")
print(f"Destination molecule: {dst_mol}")

args_dir = '~'
os.chdir(args_dir)

if "h2o" in src_file or "water" in src_file or "wat" in src_file:
    src_mol = "water"
elif "ethanol" in src_file or "eth" in src_file:
    src_mol = "ethanol"
elif "mda-enol" in src_file or "mda" in src_file:
    src_mol = "mda-enol"
elif "uracil" in src_file or "ura" in src_file:
    src_mol = "uracil"
else:
    raise ValueError(f"Unknown molecule in source file: {src_file}")

src_mol_abbr = src_mol[:3]  # Get the first three letters of the source molecule
dst_mol_abbr = dst_mol[:3]  # Get the first three letters of the destination molecule

with open(src_file, 'r') as f:
    lines = f.readlines()
    new_lines = []

    for line in lines:

        print(f"Processing line: {line.strip()}")

        if "job-name" in line:
            new_lines.append(line.replace(src_mol_abbr, dst_mol_abbr))

        elif line.startswith("apptainer"):
            if src_mol == "water":
                new_lines.append(line.replace("h2o", "water").replace(src_mol, dst_mol))
            else:
                new_lines.append(line.replace(src_mol, dst_mol))

        else:
            new_lines.append(line)

        # print(f"Updated line: {line.strip()}")
        

    new_file = src_file.replace(src_mol_abbr, dst_mol_abbr)
    with open(new_file, 'w') as new_f:
        print(f"Writing to new file: {new_file}")
        new_f.writelines(new_lines)

