import hashlib

import numpy as np
from scipy import sparse

from ..base_descriptor import BaseDescriptor
from ..constants import ATOM_SYMBOL_TO_INDEX_DICT
from ..util import _gen_2Darray_for_ffi, list_symbols_to_indices, get_hash
from ._libsymf import ffi, lib

import os
import h5py
from tqdm import tqdm

class GaussianSpecific():
    def __init__(self, Gs, elements, cutoff_func="cosine", gamma=None):
        super().__init__()
        self.fp_database = "processed/descriptors/"
        self.descriptor_type = "Gaussian"
        self.Gs = Gs
        self.elements = elements
        self.cutoff_func = cutoff_func.lower()
        if self.cutoff_func not in ["cosine", "polynomial"]:
            raise ValueError('cutoff function must be either "cosine" or "polynomial"')
        if self.cutoff_func == "polynomial":
            if gamma is None:
                raise ValueError(
                    "polynomial cutoff function requires float value > 0. of `gamma`"
                )
            elif gamma <= 0.0:
                raise ValueError("polynomial cutoff function gamma must be > 0.")
        self.gamma = gamma
        self.element_indices = list_symbols_to_indices(elements)

        self.prepare_descriptor_parameters()
        self.get_descriptor_setup_hash()

    def prepare_descriptor_parameters(self):
        self.descriptor_setup = {}
        for element in self.elements:
            if element in self.Gs:
                self.descriptor_setup[
                    element
                ] = self._prepare_descriptor_parameters_element(
                    self.Gs[element], self.element_indices
                )
            elif "default" in self.Gs:
                self.descriptor_setup[
                    element
                ] = self._prepare_descriptor_parameters_element(
                    self.Gs["default"], self.element_indices
                )
            else:
                raise NotImplementedError(
                    "Symmetry function parameters not defined properly"
                )

        self.params_set = dict()
        for element in self.elements:
            element_index = ATOM_SYMBOL_TO_INDEX_DICT[element]
            self.params_set[element_index] = dict()
            params_i = np.asarray(
                self.descriptor_setup[element][:, :3].copy(), dtype=np.intc, order="C"
            )
            params_d = np.asarray(
                self.descriptor_setup[element][:, 3:].copy(),
                dtype=np.float64,
                order="C",
            )
            self.params_set[element_index]["i"] = params_i
            self.params_set[element_index]["d"] = params_d
            self.params_set[element_index]["ip"] = _gen_2Darray_for_ffi(
                self.params_set[element_index]["i"], ffi, "int"
            )
            self.params_set[element_index]["dp"] = _gen_2Darray_for_ffi(
                self.params_set[element_index]["d"], ffi
            )
            self.params_set[element_index]["total"] = np.concatenate(
                (
                    self.params_set[element_index]["i"],
                    self.params_set[element_index]["d"],
                ),
                axis=1,
            )
            self.params_set[element_index]["num"] = len(self.descriptor_setup[element])

        return

    def _prepare_descriptor_parameters_element(self, Gs, element_indices):
        descriptor_setup = []
        cutoff = Gs["cutoff"]
        if "G2" in Gs:
            descriptor_setup += [
                [2, element1, 0, cutoff, eta, rs, 0.0]
                for eta in np.array(Gs["G2"]["etas"]) / Gs["cutoff"] ** 2
                for rs in Gs["G2"]["rs_s"]
                for element1 in element_indices
            ]

        if "G4" in Gs:
            descriptor_setup += [
                [4, element_indices[i], element_indices[j], cutoff, eta, zeta, gamma]
                for eta in np.array(Gs["G4"]["etas"]) / Gs["cutoff"] ** 2
                for zeta in Gs["G4"]["zetas"]
                for gamma in Gs["G4"]["gammas"]
                for i in range(len(element_indices))
                for j in range(i, len(element_indices))
            ]

        if "G5" in Gs:
            descriptor_setup += [
                [5, element_indices[i], element_indices[j], cutoff, eta, zeta, gamma]
                for eta in Gs["G5"]["etas"]
                for zeta in Gs["G5"]["zetas"]
                for gamma in Gs["G5"]["gammas"]
                for i in range(len(element_indices))
                for j in range(i, len(element_indices))
            ]
        return np.array(descriptor_setup)

    def get_descriptor_setup_hash(self):
        string = (
            "cosine" if self.cutoff_func == "cosine" else "polynomial%.15f" % self.gamma
        )
        for element in self.descriptor_setup.keys():
            string += element
            for desc in self.descriptor_setup[element]:
                for num in desc:
                    string += "%.15f" % num
        md5 = hashlib.md5(string.encode("utf-8"))
        hash_result = md5.hexdigest()
        self.descriptor_setup_hash = hash_result

    def save_descriptor_setup(self, filename):
        with open(filename, "w") as out_file:
            for element in self.descriptor_setup.keys():
                out_file.write(
                    "===========\nElement: {} \t num_desc: {}\n".format(
                        element, len(self.descriptor_setup[element])
                    )
                )
                for desc in self.descriptor_setup[element]:
                    out_file.write(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                            int(desc[0]),
                            int(desc[1]),
                            int(desc[2]),
                            desc[3],
                            desc[4],
                            desc[5],
                            desc[6],
                        )
                    )

    def calculate_fingerprints(self, atoms, element, cal_atoms, calc_derivatives, log):
        element_index = ATOM_SYMBOL_TO_INDEX_DICT[element]

        symbols = np.array(atoms.get_chemical_symbols())
        atom_num = len(symbols)
        atom_indices = list_symbols_to_indices(symbols)
        unique_atom_indices = np.unique(atom_indices)

        type_num = dict()
        type_idx = dict()

        cal_index = []
        for c_e in cal_atoms.keys():
            cal_index.extend(cal_atoms[c_e])
        cal_col_index = []
        for index in cal_index:
            cal_col_index.extend([index * 3, index * 3+1, index*3+2])
            
        for atom_index in unique_atom_indices:
            tmp = atom_indices == atom_index
            type_num[atom_index] = np.sum(tmp).astype(np.int64)
            # if atom indexs are sorted by atom type,
            # indexs are sorted in this part.
            # if not, it could generate bug in training process for force training
            type_idx[atom_index] = np.arange(atom_num)[tmp]

        atom_indices_p = ffi.cast("int *", atom_indices.ctypes.data)

        cart = np.copy(atoms.get_positions(wrap=True), order="C")
        scale = np.copy(atoms.get_scaled_positions(), order="C")
        cell = np.copy(atoms.cell, order="C")
        pbc = np.copy(atoms.get_pbc()).astype(np.intc)

        cart_p = _gen_2Darray_for_ffi(cart, ffi)
        scale_p = _gen_2Darray_for_ffi(scale, ffi)
        cell_p = _gen_2Darray_for_ffi(cell, ffi)
        pbc_p = ffi.cast("int *", pbc.ctypes.data)

        calc_atoms = np.asarray(cal_atoms[element], dtype=np.intc, order="C")
        cal_num = len(calc_atoms)
        cal_atoms_p = ffi.cast("int *", calc_atoms.ctypes.data)

        size_info = np.array([atom_num, cal_num, self.params_set[element_index]["num"]])

        if calc_derivatives:
            x = np.zeros(
                [cal_num, self.params_set[element_index]["num"]],
                dtype=np.float64,
                order="C",
            )
            dx = np.zeros(
                [cal_num * self.params_set[element_index]["num"], atom_num * 3],
                dtype=np.float64,
                order="C",
            )

            x_p = _gen_2Darray_for_ffi(x, ffi)
            dx_p = _gen_2Darray_for_ffi(dx, ffi)

            errno = (
                lib.calculate_sf_cos(
                    cell_p,
                    cart_p,
                    scale_p,
                    pbc_p,
                    atom_indices_p,
                    atom_num,
                    cal_atoms_p,
                    cal_num,
                    self.params_set[element_index]["ip"],
                    self.params_set[element_index]["dp"],
                    self.params_set[element_index]["num"],
                    x_p,
                    dx_p,
                )
                if self.cutoff_func == "cosine"
                else lib.calculate_sf_poly(
                    cell_p,
                    cart_p,
                    scale_p,
                    pbc_p,
                    atom_indices_p,
                    atom_num,
                    cal_atoms_p,
                    cal_num,
                    self.params_set[element_index]["ip"],
                    self.params_set[element_index]["dp"],
                    self.params_set[element_index]["num"],
                    x_p,
                    dx_p,
                    self.gamma,
                )
            )

            if errno == 1:
                raise NotImplementedError("Descriptor not implemented!")
            fp = np.array(x)
            fp_prime = np.array(dx)
            fp_prime = fp_prime[:, cal_col_index]

            return (
                size_info,
                fp,
                fp_prime
            )

        else:
            x = np.zeros(
                [cal_num, self.params_set[element_index]["num"]],
                dtype=np.float64,
                order="C",
            )
            x_p = _gen_2Darray_for_ffi(x, ffi)

            errno = (
                lib.calculate_sf_cos_noderiv(
                    cell_p,
                    cart_p,
                    scale_p,
                    pbc_p,
                    atom_indices_p,
                    atom_num,
                    cal_atoms_p,
                    cal_num,
                    self.params_set[element_index]["ip"],
                    self.params_set[element_index]["dp"],
                    self.params_set[element_index]["num"],
                    x_p,
                )
                if self.cutoff_func == "cosine"
                else lib.calculate_sf_poly_noderiv(
                    cell_p,
                    cart_p,
                    scale_p,
                    pbc_p,
                    atom_indices_p,
                    atom_num,
                    cal_atoms_p,
                    cal_num,
                    self.params_set[element_index]["ip"],
                    self.params_set[element_index]["dp"],
                    self.params_set[element_index]["num"],
                    x_p,
                    dx_p,
                    self.gamma,
                )
            )

            if errno == 1:
                raise NotImplementedError("Descriptor not implemented!")
            fp = np.array(x)

            return size_info, fp, None
        
    def prepare_fingerprints(
        self, images, calc_derivatives, save_fps, verbose, cores, log
    ):
        images_descriptor_list = []

        # if save is true, create directories if not exist
        self._setup_fingerprint_database(save_fps=save_fps)

        for image in tqdm(
            images,
            total=len(images),
            desc="Computing fingerprints",
            disable=not verbose,
        ):
            image_hash = get_hash(image)
            image_db_filename = "{}/{}.h5".format(self.desc_fp_database_dir, image_hash)

            # if save, then read/write from db as needed
            if save_fps:
                temp_descriptor_list = self._compute_fingerprints(
                    image,
                    image_db_filename,
                    calc_derivatives=calc_derivatives,
                    save_fps=save_fps,
                    cores=cores,
                    log=log,
                )

            # if not save, compute fps on-the-fly
            else:
                temp_descriptor_list = self._compute_fingerprints_nodb(
                    image,
                    image_db_filename,
                    calc_derivatives=calc_derivatives,
                    save_fps=save_fps,
                    cores=cores,
                    log=log,
                )

            images_descriptor_list += temp_descriptor_list

        return images_descriptor_list

    def _compute_fingerprints(
        self, image, image_db_filename, calc_derivatives, save_fps, cores, log
    ):
        descriptor_list = []

        with h5py.File(image_db_filename, "a") as db:
            image_dict = {}

            symbol_arr = np.array(image.get_chemical_symbols())
            image_dict["atomic_numbers"] = list_symbols_to_indices(symbol_arr)
            num_atoms = len(symbol_arr)
            image_dict["num_atoms"] = num_atoms

            try:
                current_snapshot_grp = db[str(0)]
            except Exception:
                current_snapshot_grp = db.create_group(str(0))

            num_desc_list = []
            index_arr_dict = {}
            num_desc_dict = {}
            fp_dict = {}

            fp_primes_dict = {}     
            cal_atoms = {}
            for element in self.elements:
                index_arr = np.arange(num_atoms)[symbol_arr == element]
                index_arr_dict[element] = index_arr
            for element in self.elements:
                cal_atoms[element] = []
                for index in index_arr_dict[element]:
                    if image[index].tag == 1:
                        cal_atoms[element].append(index)

            for element in self.elements:
                if element in image.get_chemical_symbols():
                    index_arr = np.arange(num_atoms)[symbol_arr == element]
                    index_arr_dict[element] = index_arr

                    try:
                        current_element_grp = current_snapshot_grp[element]
                    except Exception:
                        current_element_grp = current_snapshot_grp.create_group(element)

                    if calc_derivatives:
                        try:
                            size_info = np.array(current_element_grp["size_info"])
                            fps = np.array(current_element_grp["fps"])
                            fp_primes = np.array(
                                current_element_grp["fp_primes"]
                            )
                            cal_atoms[element] = np.array(
                                current_element_grp["cal_atoms"]
                            )
                        except Exception:
                            (
                                size_info,
                                fps,
                                fp_primes,
                            ) = self.calculate_fingerprints(
                                image,
                                element,
                                cal_atoms,
                                calc_derivatives=calc_derivatives,
                                log=log,
                            )

                            if save_fps:
                                current_element_grp.create_dataset(
                                    "size_info", data=size_info
                                )
                                current_element_grp.create_dataset("fps", data=fps)
                                current_element_grp.create_dataset(
                                    "fp_primes", data=fp_primes
                                )
                                current_element_grp.create_dataset(
                                    "cal_atoms", data=cal_atoms[element]
                                )                                

                        fp_dict[element] = fps
                        num_desc_list.append(size_info[2])
                        num_desc_dict[element] = size_info[2]
                        fp_primes_dict[element] = fp_primes
                    else:
                        try:
                            size_info = np.array(current_element_grp["size_info"])
                            fps = np.array(current_element_grp["fps"])
                        except Exception:
                            size_info, fps,_, = self.calculate_fingerprints(
                                image,
                                element,
                                calc_derivatives=calc_derivatives,
                                log=log,
                            )

                            if save_fps:
                                current_element_grp.create_dataset(
                                    "size_info", data=size_info
                                )
                                current_element_grp.create_dataset("fps", data=fps)
                                
                                current_element_grp.create_dataset(
                                    "cal_atoms", cal_atoms[element]
                                )
                        num_desc_list.append(size_info[2])
                        num_desc_dict[element] = size_info[2]
                        fp_dict[element] = fps

                else:
                    pass
                    # print("element not in current image: {}".format(element))

            num_desc_max = np.max(num_desc_list)
            num_atoms = 0
            for element in self.elements:
                num_atoms += len(cal_atoms[element])
            num_total_atoms = len(image)
            
            image_fp_array = np.zeros((num_atoms, num_desc_max))
            line = 0
            cal_atomic_numbers = []
            cal_atom_index = []
            
            for element in fp_dict.keys():
                if len(cal_atoms[element]) > 0:
                    image_fp_array[
                        np.arange(line,line + len(cal_atoms[element])), : num_desc_dict[element]
                    ] = fp_dict[element]
                    cal_atomic_numbers.extend([element for i in range(len(cal_atoms[element]))])
                    cal_atom_index.extend(cal_atoms[element])
                line += len(cal_atoms[element])
            image_dict["descriptors"] = image_fp_array
            image_dict["num_descriptors"] = num_desc_dict

            image_dict["atomic_numbers"] = list_symbols_to_indices(cal_atomic_numbers)
            image_dict["cal_atom_index"] = cal_atom_index
            if calc_derivatives:
                image_fp_prime_array = np.zeros((num_atoms * num_desc_max, 3 * num_atoms))
                line = 0
                for element in fp_dict.keys():
                    if len(cal_atoms[element]) > 0:
                        image_fp_prime_array[
                            np.arange(line,line + num_desc_max * len(cal_atoms[element])), : ] = fp_primes_dict[element]
                    line += num_desc_max * len(cal_atoms[element]) 
                image_dict["descriptor_primes"] = image_fp_prime_array
            descriptor_list.append(image_dict)

        return descriptor_list

    def _compute_fingerprints_nodb(
        self, image, image_db_filename, calc_derivatives, save_fps, cores, log
    ):
        descriptor_list = []

        image_dict = {}

        symbol_arr = np.array(image.get_chemical_symbols())
        image_dict["atomic_numbers"] = list_symbols_to_indices(symbol_arr)
        num_atoms = len(symbol_arr)
        image_dict["num_atoms"] = num_atoms

        num_desc_list = []
        index_arr_dict = {}
        num_desc_dict = {}
        fp_dict = {}
        fp_primes_dict = {}

        cal_atoms = {}
        for element in self.elements:
            index_arr = np.arange(num_atoms)[symbol_arr == element]
            index_arr_dict[element] = index_arr
        for element in self.elements:
            cal_atoms[element] = []
            for index in index_arr_dict[element]:
                if image[index].tag == 1:
                    cal_atoms[element].append(index)             
        for element in self.elements:
            if element in image.get_chemical_symbols():
                index_arr = np.arange(num_atoms)[symbol_arr == element]
                index_arr_dict[element] = index_arr

                if calc_derivatives:
                    (
                        size_info,
                        fps,
                        fp_primes,
                    ) = self.calculate_fingerprints(
                        image,
                        element,
                        cal_atoms,
                        calc_derivatives=calc_derivatives,
                        log=log,
                    )

                    num_desc_list.append(size_info[2])
                    num_desc_dict[element] = size_info[2]
                    fp_dict[element] = fps
                    fp_primes_dict[element] = fp_primes
                else:
                    size_info, fps,_ = self.calculate_fingerprints(
                        image, element, calc_derivatives=calc_derivatives, log=log
                    )
                    num_desc_list.append(size_info[2])
                    num_desc_dict[element] = size_info[2]
                    fp_dict[element] = fps

            else:
                pass
                # print("element not in current image: {}".format(element))

        num_desc_max = np.max(num_desc_list)
        num_atoms = 0
        for element in self.elements:
            num_atoms += len(cal_atoms[element])
        num_total_atoms = len(image)
        image_fp_array = np.zeros((num_atoms, num_desc_max))
        line = 0
        cal_atomic_numbers = []
        cal_atom_index = []
        for element in fp_dict.keys():
            if len(cal_atoms[element]) > 0:
                image_fp_array[np.arange(line,line + len(cal_atoms[element])), : num_desc_dict[element]] = fp_dict[element]
                cal_atomic_numbers.extend([element for i in range(len(cal_atoms[element]))])
                cal_atom_index.extend(cal_atoms[element])
            line += len(cal_atoms[element])
        image_dict["descriptors"] = image_fp_array
        image_dict["num_descriptors"] = num_desc_dict
        image_dict["atomic_numbers"] = list_symbols_to_indices(cal_atomic_numbers)
        image_dict["cal_atom_index"] = cal_atom_index

        if calc_derivatives:
            image_fp_prime_array = np.zeros((num_atoms * num_desc_max, 3 * num_atoms))
            line = 0
            for element in fp_dict.keys():
                if len(cal_atoms[element]) > 0:
                    image_fp_prime_array[
                        np.arange(line,line + num_desc_max * len(cal_atoms[element])), : ] = fp_primes_dict[element]
                line += num_desc_max * len(cal_atoms[element]) 
            image_dict["descriptor_primes"] = image_fp_prime_array
        descriptor_list.append(image_dict)
        return descriptor_list

    def _setup_fingerprint_database(self, save_fps):
        self.get_descriptor_setup_hash()
        self.desc_type_database_dir = "{}/{}".format(
            self.fp_database, self.descriptor_type
        )

        self.desc_fp_database_dir = "{}/{}".format(
            self.desc_type_database_dir, self.descriptor_setup_hash
        )

        if save_fps:
            os.makedirs(self.fp_database, exist_ok=True)
            os.makedirs(self.desc_type_database_dir, exist_ok=True)
            os.makedirs(self.desc_fp_database_dir, exist_ok=True)
            descriptor_setup_filename = "descriptor_log.txt"
            descriptor_setup_path = "{}/{}".format(
                self.desc_fp_database_dir, descriptor_setup_filename
            )
            self.save_descriptor_setup(descriptor_setup_path)

    def _get_element_list(self):
        return self.elements
       

