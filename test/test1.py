import ase
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator as sp
from sklearn.metrics import mean_squared_error
import torch
import numpy as np
import torch.optim as optim
import ase.io
from skorch import NeuralNetRegressor
from skorch.dataset import CVSplit
from skorch.callbacks import Checkpoint, EpochScoring
from amptorch.gaussian import SNN_Gaussian
from amptorch.model import BPNN, CustomMSELoss
from amptorch.data_preprocess import AtomsDataset, factorize_data, collate_amp, TestDataset
from amptorch.skorch_model import AMP
from amptorch.skorch_model.utils import target_extractor, energy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
images = []
#torch.manual_seed(0)
#np.random.seed(0)
image1 = ase.Atoms(
    'CuCH2',
    [
     (0, 0, 1), (1.5, 1.5, 1), (2, 2, 2),(2, 2, 0)
    ]
)
image2 = ase.Atoms(
    'CuC',
    [
     (0, 0, 0), (1.5, 1.5, 0)
    ]
)
image3 = ase.Atoms(
    'CuC2',
    [
     (0, 0, 0), (1.5, 1.5, 0), (2, 2, 2)
    ]
)

image4 = ase.Atoms(
    'CuH2',
    [
     (0, 0, 0), (1.5, 1.5, 0), (2, 2, 2)
    ]
)
image5 = ase.Atoms(
    'CuCH3',
    [
     (0, 0, 1), (1.5, 1.5, 1), (2, 2, 2), (2, 2, 0), (2, 3, 0)
    ]
)

images.append(image1)
images.append(image2)
images.append(image3)
images.append(image4)
images.append(image5)
energy = [2, 4, 6, 8, 10]
for index,image in enumerate(images):
  image.set_cell([10, 10, 10])
  for atom in image:
    if atom.symbol != 'Cu':
      atom.tag = 1
  image.wrap(pbc=[True, True, True])
  image.set_calculator(sp(atoms=image, energy=energy[index]))

Gs = {}
Gs["G2_etas"] = np.logspace(np.log10(0.05), np.log10(5.0), num=4)
Gs["G2_rs_s"] = [0] * 4
Gs["G4_etas"] = [0.005]
Gs["G4_zetas"] = [1.0]
Gs["G4_gammas"] = [+1.0, -1]
Gs["cutoff"] = 6.5
label = 'testenergy'
###############
train_split = CVSplit(cv=0.2)
args = (np.arange(len(images)),)
cv = ShuffleSplit(n_splits=10, random_state=None, test_size=0.2, train_size=None)
idx_train, idx_valid = next(iter(cv.split(*args, groups=None)))
train_image = [images[index] for index in idx_train]
images = train_image
y = [image.get_potential_energy() for image in images]
################
training_data = AtomsDataset(images, SNN_Gaussian, Gs, forcetraining=False,
                             label=label, cores=1, delta_data=None, specific_atoms=True)
for i in range(len(training_data)):
    fingerprint, energy, fprime, forces, scalings, rearange = training_data[i]
    print('image %d  energy %f' % (i, energy))
unique_atoms = training_data.elements
fp_length = training_data.fp_length
device = "cpu"
torch.set_num_threads(1)
optimizer = optim.AdamW
batch_size = len(training_data)
net = NeuralNetRegressor(
    module=BPNN(
        unique_atoms,
        [fp_length, 5, 10],
        device,
        forcetraining=False,
    ),
    criterion=CustomMSELoss,
    criterion__force_coefficient=0,
    optimizer=optimizer,
    # optimizer=torch.optim.LBFGS,
    lr=0.02,
    # lr=1e-1,
    batch_size=batch_size,
    max_epochs=30,
    iterator_train__collate_fn=collate_amp,
    iterator_train__shuffle=False,
    iterator_valid__collate_fn=collate_amp,
    iterator_valid__shuffle=False,
    device=device,
    # train_split=CVSplit(cv=0.2),
    train_split=0,
    optimizer__weight_decay=0.01,
    callbacks=[
        EpochScoring(
            energy_score,
            on_train=True,
            use_caching=True,
            target_extractor=target_extractor,
        ),
        # cp,
        # load_best_valid_loss,
        # LR_schedule
    ],
)


calc = AMP(training_data, net, label, specific_atoms=True)

calc.train(overwrite=True)
for image in images:
    image.set_calculator(calc)
    print(image.get_potential_energy())
