import os
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
import socket
from equiv_dens.training.model_loader import load_model
from equiv_dens.data.density_dataset import AtomsDensityData
import equiv_dens.utils.base as utils

import json


def cast_list_to_numbers(array):
    if isinstance(array, list):
        return [cast_list_to_numbers(elem) for elem in array]
    else:
        return float(array)


# output_values = ['energy', 'forces', 'dipole_moment']
output_values = ['energy', 'forces']
# read arguments
args, hyperparam_args = parse_command_line_arguments()

directory = args.restart  # load directory name
# load latest checkpoint
checkpoint_path = os.path.join(directory, 'checkpoints')  # checkpoint directory
checkpoint = torch.load(os.path.join(
    checkpoint_path, 'latest_checkpoint.pth'), map_location='cpu')
latest_checkpoint = checkpoint['step']
model_code = checkpoint['ID']  # load ID
for arg in vars(checkpoint['args']):
    if args.fix_arguments:
        if arg in hyperparam_args:
            print('loading hyperparam arg', arg)
            setattr(args, arg, getattr(checkpoint['args'], arg))
    else:
        print('loading all arg', arg)
        setattr(args, arg, getattr(checkpoint['args'], arg))

step = checkpoint['step']
restore = True
use_gpu = args.use_gpu and torch.cuda.is_available()
print('use gpu', use_gpu)

dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000,
                           required_properties=['energy', 'forces', 'density'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype,
                           verbose=args.verbose)

# define model
model = load_model(args, dataset)

HOST = ''
PORT = 50007
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
conn, addr = s.accept()
while(True):
    data_len = conn.recv(64)
    print('received data len', data_len)
    data_len = int(data_len.decode('ascii'))
    print('sending confirmation for data len')
    conn.sendall(('Received data len ' + str(data_len)).encode('ascii'))

    data_recv = conn.recv(data_len)
    data_in_json = data_recv.decode('ascii')
    if data_in_json == "exit":
        print('received exit command, exiting!')
        break
    data_in = json.loads(data_in_json)
    print('data_in', data_in)
    atoms = {}
    for key in data_in.keys():
        if key == 'atom_types':
            atoms['atom_numbers'] = utils.symbols_to_numbers(data_in['atom_types'])
            atoms['atom_numbers'] = torch.LongTensor(atoms['atom_numbers'])
        else:
            # data_num = [float(num) for num in data_in[key]]
            data_num = cast_list_to_numbers(data_in[key])
            atoms[key] = torch.tensor(data_num).type(args.dtype)
    # atoms['positions'] = atoms['positions'].view((-1, len(atoms['atom_numbers']), 3))
    print('positions shape', atoms['positions'].shape)
    atoms['atom_numbers'] = atoms['atom_numbers'].unsqueeze(0).repeat(len(atoms['positions']), 1)
    atoms['coords'], atoms['coord_weights'] = dataset.get_coords(atoms['positions'], data_in['atom_types'])
    print('use gpu', use_gpu)
    for key in atoms.keys():
        if use_gpu:
            atoms[key] = atoms[key].cuda()
        print('atoms ', key, 'shape:', atoms[key].shape)
        print('atoms ', key, 'type:', atoms[key].type())

    results = model(atoms)
    data_out = {key: results[key] for key in output_values}
    for key in data_out.keys():
        data_out[key] = data_out[key].detach().cpu().numpy().tolist()
    data_out_json = json.dumps(data_out)
    print('output converted to json')
    data_len = str(len(data_out_json.encode('ascii')))
    print('output length', data_len)
    conn.sendall(data_len.encode('ascii'))
    client_resp = conn.recv(64)
    client_resp = client_resp.decode('ascii')
    print('client_response:', client_resp)
    conn.sendall(data_out_json.encode('ascii'))
    print('Sent output data')

s.close()
