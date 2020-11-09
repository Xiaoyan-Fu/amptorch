import pickle
import os
import glob
import lmdb
import ase.io
from tqdm import tqdm
from amptorch.preprocessing import AtomsToData, FeatureScaler, TargetScaler
from amptorch.descriptor.GaussianSpecific import GaussianSpecific
import random


def construct_lmdb(paths, elements, Gs, lmdb_path="./data.lmdb"):
    
    """
    data_dir: Directory containing traj files to construct dataset from
    lmdb_path: Path to store LMDB dataset
    """
    db = lmdb.open(
        lmdb_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # Define symmetry functions

    descriptor = GaussianSpecific(Gs=Gs, elements=elements, cutoff_func="Cosine")
    descriptor_setup = ("gaussianspecific", Gs, elements, {"cutoff_func": "Cosine"})
    scaling = {"type": "normalize", "range": (0, 1)}
    forcetraining = True

    a2d = AtomsToData(
        descriptor=descriptor,
        r_energy=True,
        r_forces=True,
        save_fps=False,
        fprimes=forcetraining,
    )

    data_list = []
    idx = 0
    print(paths)
    for path in tqdm(paths, desc="calc FP"):
        images = ase.io.read(path, ":")
        for image in images:
            do = a2d.convert(image, idx=idx)
            txn = db.begin(write=True)
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(do, protocol=-1))
            txn.commit()
            # data_list.append(do)  # suppress if hitting memory limits

            # get summary statistics for at most 20k random data objects
            # control percentage depending on your dataset size
            # default: sample point with 50% probability

            # unsuppress the following if using a very large dataset
            if random.randint(0, 100) < 30 and len(data_list) < 20000:
                data_list.append(do)
            idx += 1

    feature_scaler = NoFeatureScaler(data_list, forcetraining, scaling)
    txn = db.begin(write=True)
    txn.put("feature_scaler".encode("ascii"), pickle.dumps(feature_scaler, protocol=-1))
    txn.commit()

    target_scaler = NoTargetScaler(data_list, forcetraining)
    txn = db.begin(write=True)
    txn.put("target_scaler".encode("ascii"), pickle.dumps(target_scaler, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put("elements".encode("ascii"), pickle.dumps(elements, protocol=-1))
    txn.commit()

    txn = db.begin(write=True)
    txn.put(
        "descriptor_setup".encode("ascii"), pickle.dumps(descriptor_setup, protocol=-1)
    )
    txn.commit()

    db.sync()
    db.close()

class NoFeatureScaler:
    def __init__(self, datalist, forcetraining, scaling):
        self.forcetraining = forcetraining
    def norm(self, data_list, threshold=1e-6):
        return data_list

class NoTargetScaler:
    def __init__(self, data_list, forcetraining):
        self.forcetraining = forcetraining
    def norm(self, data_list):
        return data_list
    def denorm(self, tensor, pred="energy"):
        return tensor
