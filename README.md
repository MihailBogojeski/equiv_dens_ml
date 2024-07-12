An SE3 equivariant model that predicts the electron density of a molecule.

## Installation

The easiest way to install the package is using a conda environment to take care of all the dependencies. Create using with the necessary requirements installed using:
```
$ conda create -n equiv_dens_ml python=3.10 pip
```
Next, activate the environment and install the package using:
```
$ conda activate equiv_dens_ml
$ pip install -e . -r requirements.txt

## Usage

Cofigurations for training, test and molecular dynamics runs are done via argument files stored in the 'args' directory.
There are three main scripts used for training and testing, all found in src/equiv_dens:
* train_all.py: This script is used to train multiple properties such as densities, energies and forces in multiple phases, e.g., fist the density is trained and converged, then the part of the network that predicts the density is frozen, and the energy and force models are trained using the converged density model as a base. 
* train.py: This script can be used to train multiple properties all at once. For example, if a model is to be trained on densities, energies and forces, all of the properties will be included in the loss function at once and trained
simultaneously.
* test.py: This script can be used to test a trained model given a training dataset and a set of properties.

To run a short example training and test run with a small water dataset, you can use the provided sample argument file 'args/h2o_small_all_001.txt' and the train_all.py script: 
```
python src/equiv_dens/train_all.py @args/h2o_small_all_001.txt
```
