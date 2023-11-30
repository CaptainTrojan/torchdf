import os
import time
from collections import Counter
from copy import deepcopy
from typing import Iterator, Any, Literal
from enum import Enum, IntEnum

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils import data
import pandas as pd
from torch.utils.data.dataset import T_co
import time

from tqdm import tqdm


class MultiFileHDF5ECGHandle:
    """
    Represents a loader of HDF5 ECG data from multiple files.
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        read_only: Whether to open the files as read-only or if writing is also required (for duplicate purging).
    """
    EXAM_ID_COL_NAME = 'exam_id'
    TRACINGS_COL_NAME = "tracings"
    PATIENT_COL_NAME = 'patient_id'
    
    HH_CLASS_A_COL_NAME = 'DischargeTo_Agg'


    def __init__(self, file_path, recursive=False, read_only=True):
        super().__init__()

        self.metadata_df = pd.read_csv(os.path.join(file_path, 'exams.csv'))
        
        if self.HH_CLASS_A_COL_NAME in self.metadata_df.columns:
            self.metadata_df['DischargeTo_Agg'] = self.metadata_df['DischargeTo_Agg'].fillna('Unknown')
            # self.metadata_df['DischargeTo_unit_agg'] = self.metadata_df['DischargeTo_unit_agg'].fillna('Unknown')
            
            grouped = self.metadata_df.groupby(['DischargeTo_Agg']).size().reset_index(name='Count')
            grouped.reset_index(inplace=True)
            grouped.rename(columns={'index': 'hh_class'}, inplace=True)
            
            self.metadata_df = pd.merge(self.metadata_df, grouped[['DischargeTo_Agg', 'hh_class']], 
              on=['DischargeTo_Agg'], 
              how='left')
            
        self.metadata_df.set_index(self.EXAM_ID_COL_NAME, inplace=True)
        
        self.read_only = read_only
        self.path = file_path

        # Search for all h5 files
        p = Path(file_path)
        assert (p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.hdf5'))
        else:
            files = sorted(p.glob('*.hdf5'))
        if len(files) == 0:
            raise RuntimeError('No hdf5 datasets found')

        self.__files = files

        self.__data_info_values = None
        self.__file_handle_values = None
        self.__column_values = None
        self._num_samples_total = None

    @property
    def _data_info(self):
        if self.__data_info_values is None:
            self.__setup_file_handle_structures()
        return self.__data_info_values

    @property
    def _file_handles(self):
        if self.__file_handle_values is None:
            self.__setup_file_handle_structures()
        return self.__file_handle_values

    @property
    def _columns(self):
        if self.__column_values is None:
            self.__setup_file_handle_structures()
        return self.__column_values

    def __setup_file_handle_structures(self):
        assert self.__file_handle_values is None and self.__data_info_values is None
        self.__data_info_values = []
        self.__file_handle_values = self._load_files()
        assert self.__file_handle_values is not None and self.__data_info_values is not None

    def __str__(self):
        return f"multi-file handle ({self._data_info}, " \
               f"{'unk' if self._num_samples_total is None else self._num_samples_total})"

    def __getitem__(self, index):
        """
        Fetches a sample or samples from the hdf5 files.
        :param index:
            Can be an index, slice, list of indices or a numpy array of indices
            Example:
                handle = MultiFileHDF5ECGHandle('HDF5_DATA')
                one_element = handle[54732]
                slice = handle[400:500]
                multiple_elements = handle[[5, 6, 13, 49494]]
            Returns 'tracings' by default, but if you need to fetch any other column, specify it using further tuple
            elements.
            Example:
                handle = MultiFileHDF5ECGHandle('HDF5_DATA')
                exam_ids, hashes, tracings = handle[900:1123, 'exam_id', 'hashes', 'tracings']
        :return:
        """
        if isinstance(index, tuple):
            for arg in index[1:]:
                if not isinstance(arg, str) or arg not in self._columns:
                    raise ValueError(f"{arg} is not a valid column for this dataset. Use one of {self._columns}.")

            index, *columns = index
        else:
            columns = (self.TRACINGS_COL_NAME,)

        if isinstance(index, slice):
            tensors = self.__getitem_handle_index_slice(columns, index)
        elif isinstance(index, list):
            tensors = self.__getitem_handle_index_list(columns, index)
        elif isinstance(index, np.ndarray):
            tensors = self.__getitem_handle_index_array(columns, index)
        elif isinstance(index, int):
            tensors = self.__getitem_handle_index_int(columns, index)
        else:
            raise ValueError(f"Index {index} is of an unsupported type {type(index)}. Use int, list[int] or "
                             f"numpy.ndarray of ints.")

        if len(tensors) == 1:
            return tensors[0]
        else:
            return tensors

    def __getitem_handle_index_slice(self, columns, index):
        start_file_index, start_index = self.__get_location_of_element_number(index.start)
        end_file_index, end_index = self.__get_location_of_element_number(index.stop, strict=False)
        if start_file_index == end_file_index:  # entire slice withing one file
            tensors = [self._file_handles[start_file_index][c][start_index:end_index] for c in columns]
        else:
            # first file
            tensors = []
            for c in columns:
                parts = [self._file_handles[start_file_index][c][start_index:]]

                # middle files
                for middle_file_index in range(start_file_index + 1, end_file_index):
                    parts.append(self._file_handles[middle_file_index][c][:])

                # last file
                parts.append(self._file_handles[end_file_index][c][:end_index])
                tensors.append(np.concatenate(parts, axis=0))
        return tensors

    def __getitem_handle_index_int(self, columns, index):
        start_file_index, start_index = self.__get_location_of_element_number(index)
        tensors = [self._file_handles[start_file_index][c][start_index] for c in columns]
        return tensors

    def __getitem_handle_index_array(self, columns, index):
        if not isinstance(index[0], np.integer):
            raise ValueError(f"The numpy array used for indexing must be of an integral type.")
        tensors = self.__getitem_handle_index_array_main(columns, index)
        return tensors

    def __getitem_handle_index_list(self, columns, index):
        for el in index:
            if not isinstance(el, int) and not isinstance(el, np.integer):
                raise ValueError(f"One of the indices you've used in the list, {el}, is not an integer.")
        index_array = np.array(index, dtype=int)
        tensors = self.__getitem_handle_index_array_main(columns, index_array)
        return tensors

    def __getitem_handle_index_array_main(self, columns, index_array):
        tensors = []
        for c in columns:
            parts = []
            for index in index_array:
                file_index, index = self.__get_location_of_element_number(index)
                parts.append(self._file_handles[file_index][c][index])

            tensors.append(np.stack(parts))
        return tensors

    def __len__(self):
        if self._num_samples_total is None:
            self._num_samples_total = self.get_total_amount_of_samples()
        return self._num_samples_total

    @property
    def hdf5_columns(self):
        return self._columns

    def get_total_amount_of_samples(self):
        len_metadata = len(self.metadata_df)
        len_data = sum(v['size'] for v in self._data_info)
        assert len_metadata == len_data, f"{len_metadata=} {len_data=}"
        return len_data

    def _load_files(self):
        ret = []
        total_size = 0
        expected_columns = None
        for h5dataset_fp in self.__files:
            local_columns = set()
            file_path = str(h5dataset_fp.resolve())
            file_handle = h5py.File(file_path, mode='r' if self.read_only else 'r+')
            ret.append(file_handle)
            current_size = None
            for dname, ds in file_handle.items():
                size = ds.shape[0]
                if current_size is None:
                    current_size = size
                else:
                    assert size == current_size, "Some dataset has different size than another in a file."
                local_columns.add(dname)

            if expected_columns is None:
                expected_columns = local_columns
            else:
                assert expected_columns == local_columns

            self.__data_info_values.append({'file_path': file_path,
                                            'size': current_size,
                                            'base': total_size})
            total_size += current_size

        self.__column_values = expected_columns
        assert self._columns is not None

        if self.TRACINGS_COL_NAME not in self._columns:
            raise ValueError(f"Column '{self.TRACINGS_COL_NAME}' not found in these files. Aborting.")

        if self.EXAM_ID_COL_NAME not in self._columns:
            raise ValueError(f"Column '{self.EXAM_ID_COL_NAME}' not found in these files. Aborting.")

        return ret

    def __get_location_of_element_number(self, index, strict=True):
        iterator = iter(enumerate(self._data_info))
        i, info_block = next(iterator)

        while True:
            # index exists in current file
            if index < info_block['base'] + info_block['size']:
                return i, index - info_block['base']

            try:
                i, info_block = next(iterator)
            except StopIteration:
                if strict:
                    raise ValueError(f"Index {index} out of range (last info block = {info_block})")
                else:
                    return i, index - info_block['base']

    def __collect_hash_locations(self):
        hash_locations = {}
        duplicates_found = False
        for i in tqdm(range(len(self)), desc='Reading hashes'):
            hsh = self[i, 'hashes']
            exam_id = self[i, self.EXAM_ID_COL_NAME]
            if hsh in hash_locations:
                hash_locations[hsh].append((i, exam_id))
                duplicates_found = True
            else:
                hash_locations[hsh] = [(i, exam_id)]
        return hash_locations, duplicates_found

    def check_duplicates(self):
        hash_locations, duplicates_found = self.__collect_hash_locations()

        for hsh, locations in tqdm(hash_locations.items(), desc='Verifying hash collisions'):
            anchor_tracing = self[locations[0][0], 'tracings']
            anchor_patient_id = self.metadata_df.loc[locations[0][1]]['patient_id']
            for other_location in locations[1:]:
                assert np.array_equiv(anchor_tracing, self[other_location[0], 'tracings'])

                # This doesn't apply. Clearly, duplicates could have been inserted under a wrong patient...
                # assert anchor_patient_id == self.metadata_df.loc[other_location[1]]['patient_id']

        if duplicates_found:
            print(f"Found duplicates. Overview: {Counter([len(v) for v in hash_locations.values()])}")
        else:
            print("No duplicates found.")

    def __split_index_list_and_translate(self, ordered_index_array) -> dict[int, list[int]]:
        iterator = iter(enumerate(self._data_info))
        i, info_block = next(iterator)
        ret = {}

        # for every element element
        for index in ordered_index_array:
            while index >= info_block['base'] + info_block['size']:
                try:
                    i, info_block = next(iterator)
                except StopIteration:
                    raise ValueError(f"Index {index} out of range (last info block = {info_block})")

            if i not in ret:
                ret[i] = []
            ret[i].append(index - info_block['base'])

        return ret

    def purge_duplicates(self):
        hash_locations, duplicates_found = self.__collect_hash_locations()
        to_remove_locations = []
        to_remove_exam_ids = []
        for hsh, location_list in tqdm(hash_locations.items(), desc='Selecting locations to preserve'):
            to_remove_locations += [v[0] for v in location_list[1:]]
            to_remove_exam_ids += [v[1] for v in location_list[1:]]

        translated_indices = self.__split_index_list_and_translate(sorted(to_remove_locations))
        for fid, local_indices in translated_indices.items():
            removed_index_set = set(local_indices)
            indices_to_keep = [i for i in range(self._data_info[fid]['size']) if i not in removed_index_set]

            for c in self._columns:
                dtype = self._file_handles[fid][c].dtype
                new = self._file_handles[fid][c][indices_to_keep]
                del self._file_handles[fid][c]
                self._file_handles[fid].create_dataset(c, data=new, dtype=dtype)
                print(f"File '{self._data_info[fid]['file_path']}' column '{c}' processed.")

        print("Purging csv metadata file...")
        self.metadata_df = self.metadata_df.drop(to_remove_exam_ids)
        self.metadata_df.to_csv(self.path + "/exams.csv.purged")
        print("Done.")


def scale_to_milivolts_and_add_leads(array):
    """
    V1, V2, V3, V4, V5, V6, aVL, I, aVR, II, aVF, III
    """
    ekg_matrix = np.zeros(shape=(*array.shape[:-1], 12), dtype=np.float32)
    ekg_matrix[..., 0:6] = array[..., 0:6]
    ekg_matrix[..., 7] = array[..., 6]
    ekg_matrix[..., 9] = array[..., 7]

    # aVL = I - II/2
    ekg_matrix[..., 6] = np.subtract(array[..., 6], array[..., 7] / 2)
    # aVR = -(I + II)/2
    ekg_matrix[..., 8] = -(np.add(array[..., 6], array[..., 7]) / 2)
    # aVF = II - I/2
    ekg_matrix[..., 10] = np.subtract(array[..., 7], array[..., 6] / 2)
    # II - I = III
    ekg_matrix[..., 11] = np.subtract(array[..., 7], array[..., 6])

    return torch.as_tensor(ekg_matrix, dtype=torch.float32) * 0.00488


class HDF5ECGDataset(data.IterableDataset):
    """
    Represents an ECG HDF5 dataset in the CODE-15% format not as described on PhysioNet, but as when you download it.

    :param handle: MultiHDF5FileHandle instance to load data from. If the handle is read-only, multiple datasets and
    multiple workers can make use of it to preserve memory.

    :param mode:
    How to serve data. MODE_DEFAULT = fetch unlabeled EKGs; MODE_PAIRS = fetch pairs of EKGs based on whether
    they originate from the same individual, with labels; MODE_TRIPLETS = fetch triplets of EKGs (anchor, positive,
    negative) based on whether they originate from the same individual

    :param start_fraction:
    Where to start serving data from according to the handle (default = beginning of all data)
    Can be used to implement train/test split.

    :param end_fraction:
    Where to start serving data from according to the handle (default = end of all data)
    Can be used to implement train/test split.

    :param batch_size:
    Size of served batches.

    :param num_batches_per_epoch:
    Size of epochs.

    :param seed:
    Seed for ensuring determinism. Can be used to implement test datasets.

    :param transform: PyTorch transform to apply to every data instance (default=None).

    Example:
        handle = MultiFileHDF5ECGHandle('HDF5_DATA', read_only=False)
        train_dataset = HDF5ECGDataset(handle, mode=HDF5ECGDataset.MODE_TRIPLETS, batch_size=128,
                        num_batches_per_epoch=2000, start_fraction=0.0, end_fraction=0.7)
        dev_dataset = HDF5ECGDataset(handle, mode=HDF5ECGDataset.MODE_TRIPLETS, batch_size=128,
                        num_batches_per_epoch=100, start_fraction=0.7, end_fraction=0.8, seed=17)
        test_dataset = HDF5ECGDataset(handle, mode=HDF5ECGDataset.MODE_TRIPLETS, batch_size=128,
                        num_batches_per_epoch=500, start_fraction=0.8, end_fraction=1.0, seed=17)


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=False, num_workers=10)
        ...
    """

    EXAM_ID_COL_NAME = 'exam_id'
    TRACINGS_COL_NAME = "tracings"
    PATIENT_ID_COL_NAME = 'patient_id'
    
    HH_CLASS_A_COL_NAME = 'DischargeTo_Agg'
    HH_CLASS_B_COL_NAME = 'DischargeTo_unit_agg'
    HH_CLASS_COL_NAME = 'hh_class'
    HH_VISIT_REASON_COL_NAME = 'VisitReason'

    class Mode(IntEnum):
        MODE_DEFAULT: int = 0
        MODE_PAIRS: int = 1
        MODE_TRIPLETS: int = 2
        MODE_ECG_WITH_ONE_HOT_ID: int = 3
        MODE_ECG_WITH_ID_RANDOM: int = 4
        MODE_ECG_WITH_ID_FILL: int = 5
        MODE_GALLERY_PROBE: int = 6
        MODE_MASKED_AUTOENCODER: int = 7
        MODE_HH_CLASSIFIER_SIMPLE: int = 8
        MODE_ECG_WITH_EXAM_ID: int = 9

        @classmethod
        def __contains__(cls, item):
            return item in range(0, 7)

    def __init__(self, handle: MultiFileHDF5ECGHandle, mode=Mode.MODE_DEFAULT, start_fraction=0.0, end_fraction=1.0,
                 batch_size=128, num_batches_per_epoch=200, seed=None,
                 clip_length=4096, shuffle_leads=False,
                 add_small_noise=False, transform=scale_to_milivolts_and_add_leads):
        super().__init__()

        self.handle = handle
        self.transform = transform

        if mode not in self.Mode:
            raise ValueError(f"Mode must be one of {list(HDF5ECGDataset.Mode.__members__)}.")

        self.mode = mode
        self.start_fraction = start_fraction
        self.end_fraction = end_fraction
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.seed = seed
        self.prng = None
        self.patient_groups = None
        self.indices_of_patients_with_more_than_one_sample = None
        self.clip_length = clip_length
        self.shuffle_leads = shuffle_leads
        self.add_small_noise = add_small_noise

    def lazy_init(self):
        self.patient_groups = self._build_patient_groups()
        
        if self.HH_CLASS_COL_NAME in self.handle.metadata_df.columns:
            self.hh_class_index = self._build_hh_class_inverted_index()
        
        self.indices_of_patients_with_more_than_one_sample = np.array([i for i in range(len(self.patient_groups)) if
                                                                       len(self.patient_groups[i]) > 1])

    def calculate_work_size_for_worker_id(self, num_w, wid):
        k = int(self.num_batches_per_epoch // num_w)
        remainder = self.num_batches_per_epoch - k * num_w
        if wid < remainder:
            return k + 1
        else:
            return k

    def _build_patient_groups(self):
        num_individual_samples = len(self.handle)
        start_index = int(self.start_fraction * num_individual_samples)
        end_index = int(self.end_fraction * num_individual_samples)

        ret = []
        patient_id_to_position_in_index = {}
        chunk_size = 1000
        for start_idx in range(start_index, end_index, chunk_size):
            chunk_of_exam_ids = self.handle[start_idx:min(start_idx + chunk_size, end_index), self.EXAM_ID_COL_NAME]
            corresponding_patient_ids = self.handle.metadata_df.loc[chunk_of_exam_ids][self.PATIENT_ID_COL_NAME]
            for position, pid in zip(range(start_idx, start_idx + chunk_size), corresponding_patient_ids):
                if pid not in patient_id_to_position_in_index:
                    patient_id_to_position_in_index[pid] = len(patient_id_to_position_in_index)
                    ret.append([position])
                else:
                    ret[patient_id_to_position_in_index[pid]].append(position)
        return ret
    
    def _build_hh_class_inverted_index(self):
        num_individual_samples = len(self.handle)
        start_index = int(self.start_fraction * num_individual_samples)
        end_index = int(self.end_fraction * num_individual_samples)

        ret = {}
        chunk_size = 1000
        for start_idx in range(start_index, end_index, chunk_size):
            chunk_of_exam_ids = self.handle[start_idx:min(start_idx + chunk_size, end_index), self.EXAM_ID_COL_NAME]
            corresponding_hh_classes = self.handle.metadata_df.loc[chunk_of_exam_ids][self.HH_CLASS_COL_NAME]
            # corresponding_visit_reasons = self.handle.metadata_df.loc[chunk_of_exam_ids][self.HH_VISIT_REASON_COL_NAME]
            for position, hh_class, exam_id in zip(range(start_idx, start_idx + chunk_size), corresponding_hh_classes, chunk_of_exam_ids):
                if hh_class not in ret:
                    ret[hh_class] = [(position, exam_id)]
                else:
                    ret[hh_class].append((position, exam_id))
        return ret

    def __len__(self):
        return self.num_batches_per_epoch

    def __iter__(self) -> Iterator[T_co]:
        if self.patient_groups is None:
            self.lazy_init()

        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed

        if worker_info is None:
            size = self.num_batches_per_epoch
        else:
            if seed is not None:
                seed = self.seed + worker_info.id
            size = self.calculate_work_size_for_worker_id(worker_info.num_workers, worker_info.id)
            # print(f"Worker {worker_info.id} will serve size {size} out of original {self.num_batches_per_epoch}.")

        if seed is not None:
            self.prng = np.random.default_rng(seed)
        else:
            self.prng = np.random.default_rng()

        iterator = None
        if self.mode == self.Mode.MODE_DEFAULT:
            iterator = self.__generator_default_batch_shuffled(size)
        elif self.mode == self.Mode.MODE_PAIRS:
            iterator = self.__generator_pairs(size)
        elif self.mode == self.Mode.MODE_TRIPLETS:
            iterator = self.__generator_triplets(size)
        elif self.mode == self.Mode.MODE_ECG_WITH_ONE_HOT_ID:
            iterator = self.__generator_ecg_with_one_hot_id(size)
        elif self.mode == self.Mode.MODE_ECG_WITH_ID_RANDOM:
            assert worker_info is None, "Modes 'with_id_random/fill' don't support multi-worker loading."
            iterator = self.__generator_ecg_with_id_random(size)
        elif self.mode == self.Mode.MODE_ECG_WITH_ID_FILL:
            assert worker_info is None, "Modes 'with_id_random/fill' don't support multi-worker loading."
            assert self.batch_size == 1, "Mode 'with_id_fill' does not support batch_size > 1."
            iterator = self.__generator_ecg_with_id_fill(size)
        elif self.mode == self.Mode.MODE_GALLERY_PROBE:
            assert worker_info is None, "Mode 'gallery/probe' doesn't support multi-worker loading."
            assert self.batch_size == 1, "Mode 'gallery/probe' does not support batch_size > 1."
            iterator = self.__generator_gallery_probe(size)
        elif self.mode == self.Mode.MODE_MASKED_AUTOENCODER:
            iterator = self.__generator_masked_autoencoder(size)
        elif self.mode == self.Mode.MODE_HH_CLASSIFIER_SIMPLE:
            iterator = self.__generator_hh_classifier_simple(size)
        elif self.mode == self.Mode.MODE_ECG_WITH_EXAM_ID:
            assert worker_info is None, "Mode 'ecg_with_exam_id' doesn't support multi-worker loading."
            iterator = self.__generator_ecg_with_exam_id(size)

        if iterator is None:
            raise ValueError(f"Unknown mode {self.mode}.")
        
        def apply_transform_ecg(ecg):
            if self.transform:
                ecg = self.transform(ecg)
            else:
                ecg = torch.from_numpy(ecg)
            
            if self.clip_length is not None:
                ecg = ecg[..., :self.clip_length, :]

            if self.shuffle_leads:
                idx = torch.randperm(ecg.shape[-2])
                ecg = ecg[..., idx, :]

            if self.add_small_noise:
                ecg = ecg + torch.normal(0, 0.3, size=ecg.shape, device=ecg.device)
                
            return ecg

        def apply_transform(batch_element):
            if isinstance(batch_element, tuple):
                x, y = batch_element
                if isinstance(x, dict):
                    orig_shape = x['ecg'].shape
                    x['ecg'] = apply_transform_ecg(x['ecg'])
                else:
                    orig_shape = x.shape
                    x = apply_transform_ecg(x)
                    
                if y.shape == orig_shape:
                    y = apply_transform_ecg(y)
                return x, y
            else:
                x = batch_element
                if isinstance(x, dict):
                    x['ecg'] = apply_transform_ecg(x['ecg'])
                else:
                    x = apply_transform_ecg(x)
                return x
                
        return map(apply_transform, iterator)

    def __generator_default_shuffle_batches(self, size):
        num_individual_samples = len(self.handle) - self.batch_size
        start_index = int(self.start_fraction * num_individual_samples)
        end_index = int(self.end_fraction * num_individual_samples)
        for i in range(size):
            random_index = self.prng.integers(start_index, end_index, 1)[0]
            batch = self.handle[random_index:random_index + self.batch_size]

            yield batch

    def __generator_default_batch_shuffled(self, size):
        num_individual_samples = len(self.handle)
        start_index = int(self.start_fraction * num_individual_samples)
        end_index = int(self.end_fraction * num_individual_samples)
        possibilities = np.arange(start_index, end_index)
        for i in range(size):
            random_indices = self.prng.choice(possibilities, self.batch_size, replace=False)
            batch = self.handle[random_indices]

            yield batch

    def __generator_pairs(self, size):
        # print("init...")
        num_patients = len(self.patient_groups)

        for i in range(size):
            # positive pairs
            # print("generating positive")
            positive_classes = self.prng.choice(self.indices_of_patients_with_more_than_one_sample,
                                                size=int(self.batch_size // 2) + int(self.batch_size % 2),
                                                replace=False)

            positive_pair_indices = [
                self.prng.choice(self.patient_groups[i], 2, replace=False) for i in positive_classes
            ]
            positive_pair_indices = np.array(positive_pair_indices)
            positive_pair_indices.shape = (-1,)

            # print("generating negative")
            negative_classes = self.prng.choice(np.arange(num_patients), self.batch_size,
                                                replace=False)
            negative_example_indices = np.array(
                [self.prng.choice(self.patient_groups[i], 1) for i in negative_classes]).reshape(-1)

            all_indices = np.concatenate([positive_pair_indices, negative_example_indices])
            # print("fetching...")
            corresponding_ekgs = self.handle[all_indices]

            corresponding_ekgs.shape = (corresponding_ekgs.shape[0] // 2,
                                        2,
                                        corresponding_ekgs.shape[1],
                                        corresponding_ekgs.shape[2])
            # print(corresponding_ekgs.shape)

            yield corresponding_ekgs, torch.concatenate([
                torch.ones(int(self.batch_size // 2) + int(self.batch_size % 2)),
                torch.zeros(int(self.batch_size // 2))])

    def __generator_triplets(self, size):
        num_patients = len(self.patient_groups)

        for i in range(size):
            # positive pairs
            # print("generating positive")
            positive_classes = self.prng.choice(self.indices_of_patients_with_more_than_one_sample,
                                                size=self.batch_size,
                                                replace=False)
            pc_set = set(positive_classes)
            other = [v for v in range(num_patients) if v not in pc_set]

            negative_classes = self.prng.choice(other, size=self.batch_size, replace=False)

            positive_pair_indices = np.array([
                self.prng.choice(self.patient_groups[i], 2, replace=False) for i in positive_classes
            ])
            negative_example_indices = np.array([
                self.prng.choice(self.patient_groups[i], 1) for i in negative_classes
            ])

            all_indices = np.concatenate([positive_pair_indices, negative_example_indices.reshape(-1, 1)], axis=1)
            # print("fetching...")

            original_shape = all_indices.shape
            corresponding_ekgs = self.handle[all_indices.reshape(-1)]
            corresponding_ekgs.shape = (*original_shape, corresponding_ekgs.shape[1], corresponding_ekgs.shape[2])

            yield corresponding_ekgs

    def __generator_ecg_with_one_hot_id(self, size):
        num_patients = len(self.patient_groups)

        for i in range(size):
            random_patients_with_replacement = \
                self.prng.choice(np.arange(num_patients), self.batch_size,
                                 replace=True)
            ecg_indices = np.array(
                [self.prng.choice(self.patient_groups[i], 1)
                 for i in random_patients_with_replacement]).reshape(-1)

            corresponding_ecgs = self.handle[ecg_indices]
            one_hot_patient_id = torch.zeros(self.batch_size, num_patients)
            one_hot_patient_id[torch.arange(self.batch_size), random_patients_with_replacement] = 1
            yield corresponding_ecgs, one_hot_patient_id

    def __generator_ecg_with_id_random(self, size):
        num_individual_samples = len(self.handle)
        start_index = int(self.start_fraction * num_individual_samples)
        end_index = int(self.end_fraction * num_individual_samples)
        samples = self.prng.choice(end_index - start_index, size * self.batch_size, replace=False)

        for i in range(size):
            a = i * self.batch_size
            b = (i+1) * self.batch_size
            exam_ids, ecgs = self.handle[start_index + samples[a:b], self.EXAM_ID_COL_NAME, self.TRACINGS_COL_NAME]
            corresponding_patient_ids = self.handle.metadata_df.loc[exam_ids][self.PATIENT_ID_COL_NAME].tolist()

            yield ecgs, corresponding_patient_ids

    def __generator_ecg_with_id_fill(self, size):
        ecgs_initial = []
        ecgs_rest = []
        pat_ids_initial = []
        pat_ids_rest = []

        pg_geq_2 = [i for i, pg in enumerate(self.patient_groups) if len(pg) >= 2]
        pg_eq_1 = [i for i, pg in enumerate(self.patient_groups) if len(pg) == 1]

        # Begin with patients with at least 2 ECGs
        patient_perm_geq_2 = self.prng.choice(pg_geq_2, min(len(pg_geq_2), size), replace=False)
        for pat_id, index in enumerate(patient_perm_geq_2):
            exam_ids = self.patient_groups[index]
            assert len(exam_ids) >= 2

            corresponding_ecgs = self.handle[exam_ids]
            ecgs_initial.append(corresponding_ecgs[0])
            pat_ids_initial.append(pat_id)
            for ecg in corresponding_ecgs[1:]:
                ecgs_rest.append(ecg)
                pat_ids_rest.append(pat_id)

        # If not enough, add patients with one ECG to initial set
        if size > len(pg_geq_2):
            remaining = size - len(pg_geq_2)
            offset = len(patient_perm_geq_2)
            patient_perm_eq_1 = self.prng.choice(pg_eq_1, min(remaining, len(pg_eq_1)), replace=False)
            for pat_id, index in enumerate(patient_perm_eq_1):
                exam_ids = self.patient_groups[index]
                assert len(exam_ids) == 1

                corresponding_ecgs = self.handle[exam_ids]
                ecgs_initial.append(corresponding_ecgs[0])
                pat_ids_initial.append(pat_id + offset)  # prevent ID collision

        ecgs_initial = np.array(ecgs_initial)
        ecgs_rest = np.array(ecgs_rest)
        pat_ids_initial = np.array(pat_ids_initial)
        pat_ids_rest = np.array(pat_ids_rest)

        # First, pass these two numbers as information
        yield ecgs_initial[0], len(ecgs_initial)
        yield ecgs_initial[0], len(ecgs_rest)

        # Then yield all the initial ECGs (to declare clusters)
        for i in range(len(ecgs_initial)):
            yield ecgs_initial[i:(i+1)], pat_ids_initial[i:(i+1)]

        # And finally yield ECGs to be classified in random order
        perm = self.prng.choice(len(ecgs_rest), len(ecgs_rest), replace=False)
        for i in range(len(ecgs_rest)):
            indices = perm[i * self.batch_size: (i+1) * self.batch_size]
            X = ecgs_rest[indices]
            y = pat_ids_rest[indices]
            yield X, y

    def __generator_gallery_probe(self, size):
        patient_perm = self.prng.choice(len(self.patient_groups), len(self.patient_groups), replace=False)
        skipped = 0
        S = 0
        for index in patient_perm:
            exam_ids = self.patient_groups[index]
            if len(exam_ids) < 2:
                skipped += 1
                continue

            yield np.stack(self.handle[exam_ids[:2]])
            S += 1
            if S == size:
                break
    
    def __generator_masked_autoencoder(self, size):
        num_individual_samples = len(self.handle) - self.batch_size
        start_index = int(self.start_fraction * num_individual_samples)
        end_index = int(self.end_fraction * num_individual_samples)
        for i in range(size):
            random_index = self.prng.integers(start_index, end_index, 1)[0]
            y = self.handle[random_index:random_index + self.batch_size]
            x = y.copy()

            # x shape is (batch_size, 4096, 8)
            # we want to mask various chunks across time and 3 random leads
            for sample in x:
                sample[:, self.prng.choice(8, 3, replace=False)] = 0
                sample[self.prng.choice(4096, 1000, replace=False), :] = 0

            yield x, y
            
    def __generator_hh_classifier_simple(self, size):
        forbidden_classes = [1]
        
        for i in range(size):
            num_classes = len(self.hh_class_index) - len(forbidden_classes)

            # Number of samples per class in each batch
            samples_per_class = self.batch_size // num_classes

            # Initialize empty lists to hold the batch data and labels
            batch = []
            labels = []
            visit_reasons = []
            ventricular_rates = []
            atrial_rates = []
            ages = []

            # Sample equally from each class
            for class_label, class_indices in self.hh_class_index.items():
                if class_label in forbidden_classes:
                    continue
                # Randomly choose samples from this class
                chosen_indices, exam_ids = zip(*self.prng.choice(class_indices, samples_per_class))
                chosen_indices = list(chosen_indices)
                
                # Add the chosen samples to the batch
                batch.extend(self.handle[chosen_indices])
                
                if class_label > forbidden_classes[0]:
                    class_label -= 1
                # Add the corresponding labels to the labels list
                labels.extend([class_label] * samples_per_class)
                
                metadata = self.handle.metadata_df.loc[exam_ids, ('VisitReason', 'ventricular_rate', 'atrial_rate', 'age')]
                b_reason = metadata['VisitReason']
                b_ventricular_rate = metadata['ventricular_rate']
                b_atrial_rate = metadata['atrial_rate']
                b_age = metadata['age']
                
                visit_reasons.extend(b_reason)
                ventricular_rates.extend(b_ventricular_rate)
                atrial_rates.extend(b_atrial_rate)
                ages.extend(b_age)

            # If the batch size is not a multiple of the number of classes, 
            # randomly choose extra samples from all classes to fill the batch
            if len(batch) < self.batch_size:
                chosen_classes = self.prng.choice(
                    list(i for i in self.hh_class_index.keys() if i not in forbidden_classes),
                    self.batch_size - len(batch),
                    replace=False
                )
                
                # Sample equally from each class
                for class_label in chosen_classes:
                    class_indices = self.hh_class_index[class_label]
                    
                    chosen_indices, exam_ids = zip(*self.prng.choice(class_indices, samples_per_class))
                    chosen_indices = list(chosen_indices)
                    
                    # Add the chosen samples to the batch
                    batch.extend(self.handle[chosen_indices])
                    
                    if class_label > forbidden_classes[0]:
                        class_label -= 1
                    # Add the corresponding labels to the labels list
                    labels.extend([class_label] * samples_per_class)
                    
                    metadata = self.handle.metadata_df.loc[exam_ids, ('VisitReason', 'ventricular_rate', 'atrial_rate', 'age')]
                    b_reason = metadata['VisitReason']
                    b_ventricular_rate = metadata['ventricular_rate']
                    b_atrial_rate = metadata['atrial_rate']
                    b_age = metadata['age']
                    
                    visit_reasons.extend(b_reason)
                    ventricular_rates.extend(b_ventricular_rate)
                    atrial_rates.extend(b_atrial_rate)
                    ages.extend(b_age)
            
            batch = np.array(batch)
            labels = np.array(labels)
            visit_reasons = np.array(visit_reasons)
            ventricular_rates = np.array(ventricular_rates)
            atrial_rates = np.array(atrial_rates)
            ages = np.array(ages)
            
            yield {
                'ecg': batch,
                'visit_reasons': visit_reasons,
                'ventricular_rates': ventricular_rates,
                'atrial_rates': atrial_rates,
                'ages': ages
            }, labels
    
    def __generator_ecg_with_exam_id(self, size):
        num_individual_samples = len(self.handle) - self.batch_size
        start_index = int(self.start_fraction * num_individual_samples)
        end_index = int(self.end_fraction * num_individual_samples)
        for i in range(start_index, end_index, self.batch_size):
            tracings, exam_ids = self.handle[i:i + self.batch_size, self.TRACINGS_COL_NAME, self.EXAM_ID_COL_NAME]

            yield {'ecg': tracings, 'exam_ids': exam_ids}
    


class ECGDataModule(LightningDataModule):
    """
    Lightning DataModule for ECG HDF5 datasets.
    """

    def __init__(self, hdf5_path, batch_size,
                 mode: Literal[
                     HDF5ECGDataset.Mode.MODE_DEFAULT,
                     HDF5ECGDataset.Mode.MODE_PAIRS,
                     HDF5ECGDataset.Mode.MODE_TRIPLETS,
                     HDF5ECGDataset.Mode.MODE_ECG_WITH_ONE_HOT_ID,
                 ] = HDF5ECGDataset.Mode.MODE_DEFAULT,
                 sample_size=2000,
                 num_workers=0,
                 train_fraction=0.7,
                 dev_fraction=0.1,
                 test_fraction=0.2,
                 clip_length=None,
                 shuffle_leads=False,
                 add_small_noise=False,
                 dev_seed=17,
                 test_seed=31
                 ):
        super().__init__()

        assert train_fraction + dev_fraction + test_fraction == 1.0

        self.__mode = mode
        self.__train_fraction = train_fraction
        self.__dev_fraction = dev_fraction
        self.__test_fraction = test_fraction

        # when train should be different from val/test
        self.__special_mode_mapping = {
            HDF5ECGDataset.Mode.MODE_ECG_WITH_ONE_HOT_ID: HDF5ECGDataset.Mode.MODE_PAIRS
        }

        self.__handle = MultiFileHDF5ECGHandle(hdf5_path)
        self.__train_dataset = HDF5ECGDataset(
            handle=self.__handle,
            mode=mode,
            start_fraction=0.0,
            end_fraction=self.__train_fraction,
            batch_size=batch_size,
            num_batches_per_epoch=int(sample_size * self.__train_fraction // batch_size),
            clip_length=clip_length,
            shuffle_leads=shuffle_leads,
            add_small_noise=add_small_noise
        )
        self.__dev_dataset = HDF5ECGDataset(
            handle=self.__handle,
            mode=self.__special_mode_mapping.get(mode, mode),
            start_fraction=self.__train_fraction,
            end_fraction=self.__train_fraction + self.__dev_fraction,
            batch_size=batch_size,
            num_batches_per_epoch=int(sample_size * self.__dev_fraction // batch_size),
            clip_length=clip_length,
            seed=dev_seed
        )
        self.__test_dataset = HDF5ECGDataset(
            handle=self.__handle,
            mode=self.__special_mode_mapping.get(mode, mode),
            start_fraction=self.__train_fraction + self.__dev_fraction,
            end_fraction=1.0,
            batch_size=batch_size,
            num_batches_per_epoch=int(sample_size * self.__test_fraction // batch_size),
            clip_length=clip_length,
            seed=test_seed
        )
        self.__num_workers = num_workers

    def force_open_handles(self):
        """
        Forces all file handles open.
        Shouldn't be used before 'fork' (risking race conditions)
        nor 'spawn' (pickle crash).
        :return:
        """
        self.__train_dataset.lazy_init()
        self.__test_dataset.lazy_init()
        self.__dev_dataset.lazy_init()

    @property
    def patient_counts(self):
        """
        :return: A 3-tuple of patient counts (one-hot-embedding size); 1 value for each dataset from the
        train/dev/test split. Example: (25976, 7171, 11726)
        """
        if self.__mode == HDF5ECGDataset.Mode.MODE_ECG_WITH_ONE_HOT_ID:
            if self.__train_dataset.patient_groups is None:
                raise ValueError("Dataset has not yet been accessed, the handles are not opened. "
                                 "You can call 'force_open_handles()' on this object but cannot "
                                 "use multiprocess training afterwards.")
            return (
                len(self.__train_dataset.patient_groups),
                len(self.__dev_dataset.patient_groups),
                len(self.__test_dataset.patient_groups)
            )
        else:
            raise ValueError(f"Mode {self.__mode} does not support patient count retrieval.")

    def make_dataloader(self, ds):
        return torch.utils.data.DataLoader(ds,
                                           batch_size=None,
                                           persistent_workers=True if self.__num_workers > 0 else False,
                                           shuffle=False, num_workers=self.__num_workers, pin_memory=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.make_dataloader(self.__train_dataset)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.make_dataloader(self.__dev_dataset)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.make_dataloader(self.__test_dataset)
    
if __name__ == '__main__':
    dh = ECGDataModule('datasets/hhmusedata', batch_size=32, mode=HDF5ECGDataset.Mode.MODE_HH_CLASSIFIER_SIMPLE,
                       train_fraction=0.7,
                       dev_fraction=0.1,
                       test_fraction=0.2,)
    num_train_batches = len(dh.train_dataloader())
    num_dev_batches = len(dh.val_dataloader())
    
    train_class_counts = np.zeros(5)
    dev_class_counts = np.zeros(5)
    for batch in tqdm(dh.train_dataloader(), desc='Counting classes'):
        _, y = batch
        train_class_counts += np.bincount(y, minlength=5)
    for batch in tqdm(dh.val_dataloader(), desc='Counting classes'):
        _, y = batch
        dev_class_counts += np.bincount(y, minlength=5)
    
    print(f"Train batches: {num_train_batches}")
    print(f"Dev batches: {num_dev_batches}")
    print(f"Train class counts: {train_class_counts}")
    print(f"Dev class counts: {dev_class_counts}")
    