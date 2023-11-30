import multiprocessing
from multiprocessing import Lock
import threading
from pathlib import Path
from time import time, sleep

import numpy as np
from tqdm import tqdm

import os

from libs.pacient_ekg import PatientEKGFile
from itertools import repeat
import h5py
from collections import deque
import hashlib



class IncrementAndReturn:
    def __init__(self):
        self.__value = 0
        self.__lock = Lock()

    def get_id(self):
        with self.__lock:
            self.__value += 1
            return self.__value


def threaded_progress_bar(description, eta_window=5):
    times = deque(maxlen=eta_window)
    results = deque(maxlen=1)

    def decorator(func):
        def wrapper(*args, **kwargs):
            def run_func():
                start_time = time()
                results.append(func(*args, **kwargs))
                end_time = time()
                times.append(end_time - start_time)

            if len(times) > 0:
                estimated_total_time = sum(times) / len(times)

                t = threading.Thread(target=run_func)
                t.start()
                bar = tqdm(range(int(estimated_total_time)), desc=description, leave=True)
                for _ in bar:
                    if not t.is_alive():
                        break
                    sleep(1)
                else:
                    bar.set_description("Finishing...")

                t.join()
            else:
                print("No time to estimate, first dry run...")
                run_func()

            return results[0]

        return wrapper

    return decorator


def walk_all_files_queue(root):
    ret = []
    for d, dr, fl in os.walk(root):
        if len(fl) == 0:
            continue
        for f in fl:
            ret.append(os.path.join(d, f))
    return ret


def create_one_hdf5_part(OUTPUT_DIRECTORY, LIMIT_PER_FILE, BURST_SIZE, BLOCK_SIZE,
                         file_name_list: list[str], file_name_index: multiprocessing.Value,
                         results: multiprocessing.Queue,
                         id_source):
    
    annotations = {}
    import csv
    with open('annotations_clean.csv', 'r', encoding='utf-8') as f:
        csv_parser = csv.reader(f, delimiter=';')
        next(csv_parser)
        for line in csv_parser:
            *values, key = line
            assert len(values) == 18
            annotations[key] = values
        
    id_path_map = open(f'{OUTPUT_DIRECTORY}/id_path_map.csv', 'a')
    with open(Path(f'{OUTPUT_DIRECTORY}/exams.csv'), 'a', encoding='utf-8') as exams_file:
        all_cardiograms = []
        bar = tqdm(total=LIMIT_PER_FILE, desc=f"Building part", leave=False)
        ret = {'fail_counter': 0}
        pool = multiprocessing.Pool()
        num_processed_files = 0

        while len(all_cardiograms) < LIMIT_PER_FILE and file_name_index.value < len(file_name_list):
            current_file_name_burst = []

            with file_name_index.get_lock():
                for _ in range(BURST_SIZE):
                    current_file_name_burst.append(file_name_list[file_name_index.value])
                    file_name_index.value += 1
                    if file_name_index.value == len(file_name_list):
                        break

            old_size = len(all_cardiograms)
            num_processed_files += len(current_file_name_burst)
            for result in pool.imap_unordered(work, zip(current_file_name_burst, repeat(BLOCK_SIZE))):
                if result is not None:
                    # first_dimension_distribution.append(result[0])
                    # second_dimension_distribution.append(result[1])
                    add_cardiogram(all_cardiograms, result, id_source, exams_file, id_path_map, annotations)
                else:
                    ret['fail_counter'] += 1
            new_size = min(len(all_cardiograms), LIMIT_PER_FILE)
            bar.update(new_size - old_size)
        pool.terminate()

        # print("Compressing hdf5...")
        bar.set_description("Compressing hdf5...")
        dump_to_hdf5(all_cardiograms, OUTPUT_DIRECTORY)
        ret['status'] = 'success'
        ret['msg'] = f'iteration complete, {num_processed_files=}'
        results.put(ret)
        bar.set_description("Completed.")
        id_path_map.close()
        return 0


def add_cardiogram(all_cardiograms, result, id_source, exams_file, id_path_map, annotations):
    full_path, ecg, tail_size, *metadata = result
    file_name = os.path.basename(full_path)
    try:
        metadata += annotations[file_name]
    except KeyError:
        return
    assert len(metadata) == 26, "it is actually " + str(len(metadata)) + " " + str(metadata)
    add_one_ecg_to_dataset(all_cardiograms, full_path, ecg, exams_file, id_path_map, id_source, metadata, tail_size)


def add_one_ecg_to_dataset(all_cardiograms, full_path, ecg, exams_file, id_path_map, id_source, metadata, size):
    with id_source.get_lock():
        id_source.value += 1
        exam_id = id_source.value
    values = [exam_id, *metadata]
    for i in range(len(values)):
        if type(values[i]) == str:
            values[i] = f"\"{values[i]}\""
    assert len(values) == 27, "it is actually " + str(len(values)) + " " + str(values)
    exams_file.write(",".join(str(v) for v in values) + "\n")
    id_path_map.write(f"{exam_id},{full_path}\n")
    chunk_hash = hashlib.sha1(ecg.data.tobytes()).hexdigest()
    all_cardiograms.append((exam_id, chunk_hash, size, ecg))


# @threaded_progress_bar("Saving and compressing", eta_window=5)
def dump_to_hdf5(all_cardiograms, OUTPUT_DIRECTORY):
    part_counter = len(os.listdir(OUTPUT_DIRECTORY))

    ids, hashes, sizes, all_chunks = tuple(zip(*all_cardiograms))
    with h5py.File(Path(f'{OUTPUT_DIRECTORY}/exams_part_{part_counter}.hdf5'), 'w', libver='latest') as hf:
        hf.create_dataset("exam_id", data=ids, dtype='int32', chunks=True,)# compression='lzf', compression_opts=None)
        hf.create_dataset("hashes", data=hashes, chunks=True, compression='lzf',)# compression_opts=None)
        hf.create_dataset("real_lengths", data=sizes, dtype='int16', chunks=True,)# compression='lzf', compression_opts=None)
        hf.create_dataset("tracings", data=np.array(all_chunks), chunks=(1, 4096, 8),)# compression='lzf', compression_opts=None)


def work(args):
    full_path, BLOCK_SIZE = args
    pac_ekg = PatientEKGFile()
    try:
        pac_ekg.load_data(full_path)
    except Exception as e:
        # If the EKG has invalid structure, skip it.
        # print(full_path, e)
        return None

    if pac_ekg.get_shape() is None:
        # print(full_path, "bad shape")
        return None

    unique_id = pac_ekg.get_unique_identifier()
    gender = str(pac_ekg.gender).lower()
    if gender == 'none':
        is_male = -1
    elif gender == 'male':
        is_male = 1
    else:
        is_male = 0

    age = pac_ekg.age if pac_ekg.age is not None else -1
    ventricular_rate = int(pac_ekg.ekg_mesurements['VentricularRate']) \
        if pac_ekg.ekg_mesurements is not None and \
           'VentricularRate' in pac_ekg.ekg_mesurements \
        else -1
    atrial_rate = int(pac_ekg.ekg_mesurements['AtrialRate']) \
        if pac_ekg.ekg_mesurements is not None and \
           'AtrialRate' in pac_ekg.ekg_mesurements \
        else -1
    weight = int(pac_ekg.weight) if pac_ekg.weight is not None else -1
    height = int(pac_ekg.height) if pac_ekg.height is not None else -1
    acquisition_date = str(pac_ekg.acquisition_date) if pac_ekg.acquisition_date is not None else '<unknown>'

    np_mat = pac_ekg.gen_tensor_matrix(False)  # (8, 5000)
    if np_mat.shape[1] < BLOCK_SIZE:
        true_size = np_mat.shape[1]
        pad_size = BLOCK_SIZE - true_size
        np_mat = np.pad(np_mat, ((0, 0), (0, pad_size)))
    elif np_mat.shape[1] > BLOCK_SIZE:
        cut_size = np_mat.shape[1] - BLOCK_SIZE
        start_cut_size = cut_size // 2
        end_cut_size = cut_size // 2 + cut_size % 2
        np_mat = np_mat[:, start_cut_size:-end_cut_size]  # cut 1s from front and
        true_size = BLOCK_SIZE
    else:
        true_size = BLOCK_SIZE

    assert np_mat.shape[1] == BLOCK_SIZE
    np_mat = np_mat.T
    # np_mat /= 2 ** 15 * 0.001 * 4.88

    return full_path, np_mat, true_size, acquisition_date, unique_id, age, is_male, weight, height, ventricular_rate, atrial_rate