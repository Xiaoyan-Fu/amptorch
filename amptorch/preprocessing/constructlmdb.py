import pickle
import os
import glob
import lmdb
import ase.io
from tqdm import tqdm
from amptorch.preprocessing import AtomsToData, FeatureScaler, TargetScaler
from amptorch.descriptor.GaussianSpecific import GaussianSpecific
import random


def construct_lmdb(paths, elements, Gs, lmdb_path="./datalmdb"):
    
    """
    data_dir: Directory containing traj files to construct dataset from
    lmdb_path: Path to store LMDB dataset
    """
    lmdb_path = lmdb_path.strip()
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    configdb = lmdb.open(
        os.path.join(lmdb_path,'config.lmdb'),
        map_size=1073741824 * 1,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # paths = glob.glob(os.path.join(data_dir, "*.traj"))
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
            dbname = 'data' + str(idx) + '.lmdb'
            db = lmdb.open(
                os.path.join(lmdb_path,dbname),
                map_size=1073741824 * 1,
                subdir=False,
                meminit=False,
                map_async=True,
                )
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
            db.sync()
            db.close()

    db = configdb
    
    feature_scaler = FeatureScaler(data_list, forcetraining, scaling)
    txn = db.begin(write=True)
    txn.put("feature_scaler".encode("ascii"), pickle.dumps(feature_scaler, protocol=-1))
    txn.commit()

    target_scaler = TargetScaler(data_list, forcetraining)
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
    # norm fp and target, save in db
    for i in tqdm(range(idx), desc="Norm fp and target"):
        pathi = os.path.join(lmdb_path, 'data' + str(i) + '.lmdb') 
        env = lmdb.open(
            pathi,
            subdir=False,
            lock=False,
            readahead=False,
            map_size=1073741824 * 1,
        )
        txn = env.begin(write=True)
        data = txn.get(f"{i}".encode("ascii"))
        data_object = pickle.loads(data)
        feature_scaler.norm([data_object])
        target_scaler.norm([data_object])
        txn.put(f"{i}".encode("ascii"), pickle.dumps(data_object, protocol=-1))
        txn.commit()
        env.sync()
        env.close()