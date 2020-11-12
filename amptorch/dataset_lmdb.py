import lmdb
import pickle
from torch.utils.data import Dataset
from amptorch.descriptor.Gaussian import Gaussian
from amptorch.descriptor.GaussianSpecific import GaussianSpecific
from amptorch.descriptor.MCSH import AtomisticMCSH
import os

class AtomsLMDBDataset(Dataset):
    def __init__(
        self,
        db_path,
    ):
        self.db_path = db_path
        self.configpath = os.path.join(self.db_path, 'config.lmdb')
        env = self.connect_db(self.configpath)
        self.length = pickle.loads(env.begin().get("length".encode("ascii")))
        
        self.datapathlist = [os.path.join(self.db_path, 'data' + str(idx) + '.lmdb') for idx in range(self.length)] 
        # self.keys = [f"{j}".encode("ascii") for j in range(env.stat()["entries"])]
        self.keys = [f"{j}".encode("ascii") for j in range(self.length)]
        self.feature_scaler = pickle.loads(
            env.begin().get("feature_scaler".encode("ascii"))
        )
        self.target_scaler = pickle.loads(
            env.begin().get("target_scaler".encode("ascii"))
        )
        
        self.elements = pickle.loads(env.begin().get("elements".encode("ascii")))
        self.descriptor = self.get_descriptor(
            pickle.loads(env.begin().get("descriptor_setup".encode("ascii")))
        )
        env.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        env = self.connect_db(self.datapathlist[idx])
        data = env.begin().get(self.keys[idx])
        data_object = pickle.loads(data)
        env.close()

        return data_object

    def get_descriptor(self, descriptor_setup):
        fp_scheme, fp_params, elements, cutoff_params = descriptor_setup
        if fp_scheme == "gaussian":
            descriptor = Gaussian(Gs=fp_params, elements=elements, **cutoff_params)
        elif fp_scheme == "mcsh":
            descriptor = AtomisticMCSH(MCSHs=fp_params, elements=elements)
        elif fp_scheme == "gaussianspecific":
            descriptor = GaussianSpecific(Gs=fp_params, elements=elements, **cutoff_params)
        else:
            raise NotImplementedError
        return descriptor

    def input_dim(self):
        return self[0].fingerprint.shape[1]

    def connect_db(self, lmdb_path):
        mapsize = 1048576 * 2
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            map_size=mapsize,
        )
        return env
