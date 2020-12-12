import os
from abc import ABC, abstractmethod

import h5py
import numpy as np
from tqdm import tqdm

from .util import get_hash, list_symbols_to_indices


class BaseDescriptor(ABC):
    def __init__(self):
        super().__init__()
        self.fp_database = "processed/descriptors/"

        # To Be specified/calculated
        self.descriptor_type = "default"
        self.descriptor_setup_hash = "default"

        self.elements = []

    @abstractmethod
    def calculate_fingerprints(self, image, params_set, calculate_derivatives=True):
        # image is a single snapshot
        pass

    @abstractmethod
    def get_descriptor_setup_hash(self):
        # set self.descriptor_setup_hash
        pass

    @abstractmethod
    def save_descriptor_setup(self, filename):
        pass

    @abstractmethod
    def prepare_descriptor_parameters(self):
        # prepare self.params_set
        pass

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
                        except Exception:
                            (
                                size_info,
                                fps,
                                fp_primes,
                            ) = self.calculate_fingerprints(
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
                                    "fp_primes", data=fp_primes
                                )
                                
                        num_desc_list.append(size_info[2])
                        num_desc_dict[element] = size_info[2]
                        fp_dict[element] = fps

                        fp_primes_dict[element] = fp_primes

                    else:
                        try:
                            size_info = np.array(current_element_grp["size_info"])
                            fps = np.array(current_element_grp["fps"])
                        except Exception:
                            size_info, fps, _, = self.calculate_fingerprints(
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

                        num_desc_list.append(size_info[2])
                        num_desc_dict[element] = size_info[2]
                        fp_dict[element] = fps

                else:
                    pass
                    # print("element not in current image: {}".format(element))

            num_desc_max = np.max(num_desc_list)
            image_fp_array = np.zeros((num_atoms, num_desc_max))
            for element in fp_dict.keys():
                image_fp_array[
                    index_arr_dict[element], : num_desc_dict[element]
                ] = fp_dict[element]

            image_dict["descriptors"] = image_fp_array
            image_dict["num_descriptors"] = num_desc_dict

            if calc_derivatives:
                image_fp_prime_array = np.zeros((num_atoms * num_desc_max, 3 * num_atoms))
                for element in fp_dict.keys():
                    indexlist = []
                    for index in index_arr_dict[element]:
                        indexlist += [i for i in range(index, index+num_desc_max)]
                    image_fp_prime_array[
                        indexlist, : ] = fp_primes_dict[element]
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
                        calc_derivatives=calc_derivatives,
                        log=log,
                    )

                    num_desc_list.append(size_info[2])
                    num_desc_dict[element] = size_info[2]
                    fp_dict[element] = fps

                    fp_primes_dict[element] = fp_primes

                else:

                    size_info, fps, _, = self.calculate_fingerprints(
                        image, element, calc_derivatives=calc_derivatives, log=log
                    )

                    num_desc_list.append(size_info[2])
                    num_desc_dict[element] = size_info[2]
                    fp_dict[element] = fps

            else:
                pass
                # print("element not in current image: {}".format(element))

        num_desc_max = np.max(num_desc_list)
        image_fp_array = np.zeros((num_atoms, num_desc_max))
        for element in fp_dict.keys():
            image_fp_array[index_arr_dict[element], : num_desc_dict[element]] = fp_dict[
                element
            ]

        image_dict["descriptors"] = image_fp_array
        image_dict["num_descriptors"] = num_desc_dict

        
        if calc_derivatives:
            image_fp_prime_array = np.zeros((num_atoms * num_desc_max, 3 * num_atoms))
            for element in fp_dict.keys():
                indexlist = []
                for index in index_arr_dict[element]:
                    indexlist += [i for i in range(index, index+num_desc_max)]
                image_fp_prime_array[
                    indexlist, : ] = fp_primes_dict[element]
            image_dict["descriptor_primes"] = image_fp_prime_array
        descriptor_list.append(image_dict)
        return descriptor_list

    def _fp_prime_element_row_index_to_image_row_index(
        self, original_rows, index_arr, num_desc, num_desc_max
    ):
        atom_indices_for_specific_element, desc_indices = np.divmod(
            original_rows, num_desc
        )

        atom_indices_in_image = index_arr[atom_indices_for_specific_element]

        new_row = atom_indices_in_image * num_desc_max + desc_indices
        return new_row

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
