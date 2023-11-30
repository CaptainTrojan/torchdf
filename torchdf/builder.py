import numpy as np
from tqdm import tqdm
import pandas as pd
import wfdb
import os
import re
import signal
import h5py

import multiprocessing
import signal
from multiprocessing import Process
from queue import Empty

import numpy as np
import psutil
from tqdm import tqdm

import os

import matplotlib.pyplot as plt
import h5py
from builder_utils import walk_all_files_queue, create_one_hdf5_part



def build_ptbxl(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(f'{input_dir}/ptbxl_database.csv')
    df.rename(columns={'ecg_id': 'exam_id', 'sex': 'is_male'}, inplace=True)
    df['is_male'] = df['is_male'].replace({0: 1, 1: 0})
    df['patient_id'] = df['patient_id'].astype(int)
    df.set_index('exam_id', inplace=True)

    exam_ids_buffer = []
    tracings_buffer = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        data = wfdb.rdsamp(f'{input_dir}/' + row['filename_hr'])
        array = data[0]
        new_array = np.zeros((4096, 12), dtype=np.short)

        assert array.shape[0] == 5000
        clip_size = (5000 - 4096) // 2
        clipped = array[clip_size:-clip_size]
        new_array[:, 0:6] = clipped[:, 6:12] / 0.00488
        new_array[:, 6:8] = clipped[:, 0:2] / 0.00488
        tracings_buffer.append(new_array)
        exam_ids_buffer.append(idx)

    with h5py.File(f'{output_dir}/exams_part_0.hdf5', 'w') as f:
        f.create_dataset('exam_id', data=exam_ids_buffer, dtype='i4')
        f.create_dataset('tracings', data=tracings_buffer, chunks=(1, 4096, 8), dtype='i2')

    df.to_csv(f'{output_dir}/exams.csv')
    print('done')


def build_ptb(input_dir, output_dir):

    os.makedirs(f'{output_dir}', exist_ok=True)

    exam_ids_buffer = []
    patient_ids_buffer = []
    tracings_buffer = []

    with open(f'{input_dir}/RECORDS', 'r') as f:
        for idx, p in enumerate(f):
            p = p.strip()

            pattern = r"patient(\d+)"
            match = re.search(pattern, p)
            if match:
                patient_id = int(match.group(1))
            else:
                raise ValueError("Patient ID couldn't have been extracted.")

            path = os.path.join(f"{input_dir}", p)
            data = wfdb.rdsamp(path)

            array = data[0]
            new_array = np.zeros((4096, 12), dtype=np.short)

            clip_size = (array.shape[0] - 8192) // 2
            clipped = array[clip_size:-clip_size]
            clipped = clipped[::2, :][:4096, :]
            new_array[:, 0:6] = clipped[:, 6:12] / 0.00488
            new_array[:, 6:8] = clipped[:, 0:2] / 0.00488
            tracings_buffer.append(new_array)
            exam_ids_buffer.append(idx)
            patient_ids_buffer.append(patient_id)


    with h5py.File(f'{output_dir}/exams_part_0.hdf5', 'w') as f:
        f.create_dataset('exam_id', data=exam_ids_buffer, dtype='i4')
        f.create_dataset('tracings', data=tracings_buffer, chunks=(1, 4096, 8), dtype='i2')

    df = pd.DataFrame.from_dict({'exam_id': exam_ids_buffer, 'patient_id': patient_ids_buffer})
    df.set_index('exam_id', inplace=True)
    df.to_csv(f'{output_dir}/exams.csv')
    print('done')


def build_code15(input_dir, output_dir):
    OUT_FILE_SIZE = 50000

    exam_ids_buffer = []
    tracings_buffer = []
    K = 0

    os.mkdir(output_dir)
    df = pd.read_csv(f'{input_dir}/exams.csv')
    df.set_index('exam_id', inplace=True)
    to_remove_indices = []

    for fn in tqdm(os.listdir(input_dir), desc="Processing files"):
        path = os.path.join(input_dir, fn)

        if not path.endswith('.hdf5'):
            continue

        failed_count = 0

        with h5py.File(path) as f:
            tracings = f['tracings'][:-1]
            transformed_tracings = np.zeros((len(tracings), 4096, 8), dtype='i2')
            transformed_tracings[:, :, 0:6] = tracings[:, :, 6:12] / 0.00488
            transformed_tracings[:, :, 6:8] = tracings[:, :, 0:2] / 0.00488

            for i, (tracing, exam_id) in enumerate(zip(transformed_tracings, f['exam_id'][:-1])):
                if np.sum(tracing) == 0:
                    failed_count += 1
                    to_remove_indices.append(exam_id)
                else:
                    tracings_buffer.append(tracing)
                    exam_ids_buffer.append(exam_id)

        if len(exam_ids_buffer) >= OUT_FILE_SIZE:
            assert len(exam_ids_buffer) == len(tracings_buffer)

            to_store_exam_ids = exam_ids_buffer[:OUT_FILE_SIZE]
            to_store_tracings = tracings_buffer[:OUT_FILE_SIZE]

            exam_ids_buffer = exam_ids_buffer[OUT_FILE_SIZE:]
            tracings_buffer = tracings_buffer[OUT_FILE_SIZE:]

            with h5py.File(f'{output_dir}/exams_part_{K}.hdf5', 'w') as out:
                out.create_dataset('exam_id', shape=(len(to_store_exam_ids), ), dtype='i4', data=to_store_exam_ids)
                out.create_dataset('tracings', shape=(len(to_store_tracings), 4096, 8), dtype='i2', data=to_store_tracings,
                                   chunks=(1, 4096, 8))

            K += 1

        print(f"{failed_count=}")

    if len(exam_ids_buffer) > 0:
        with h5py.File(f'{output_dir}/exams_part_{K}.hdf5', 'w') as out:
            out.create_dataset('exam_id', shape=(len(exam_ids_buffer), ), dtype='i4', data=exam_ids_buffer)
            out.create_dataset('tracings', shape=(len(tracings_buffer), 4096, 8), dtype='i2', data=tracings_buffer)

    df = df.drop(to_remove_indices)
    df.to_csv(f'{output_dir}/exams.csv')
    print("Done.")


def handle_interrupt(sig, frame):
    global USER_WANTS_TERMINATION
    if USER_WANTS_TERMINATION:
        os.kill(os.getpid(), 9)
    else:
        print(f"Termination signal captured {signal.Signals(sig).name}")
    USER_WANTS_TERMINATION = True


def build_xml(output_dir, input_dir, limit_per_file, burst_size, block_size):
    other_keys = []
    with open('annotations_clean.csv', 'r', encoding='utf-8') as f:
        other_keys = f.readline().strip().split(";")[:-1]
    other_keys[0] = "patient_ID_2"
    other_keys_str = ",".join(other_keys)

    os.makedirs(output_dir, exist_ok=True)
    ef = open(f'{output_dir}/exams.csv', 'w')
    ef.write(f"exam_id,acquisition_date,patient_id,age,is_male,weight,height,ventricular_rate,"
             f"atrial_rate,{other_keys_str}\n")
    ef.close()
    
    signal.signal(signal.SIGINT, handle_interrupt)
    memory_history = []
    file_name_list = walk_all_files_queue(input_dir)
    results = multiprocessing.Queue()
    bar = tqdm(total=len(file_name_list), desc='All files progress')
    file_name_index = multiprocessing.Value('i', 0)
    id_source = multiprocessing.Value('i', 0)
    last_index_pos = file_name_index.value

    while not USER_WANTS_TERMINATION and file_name_index.value < len(file_name_list):
        p = Process(target=create_one_hdf5_part, name='hdf5-creator',
                    args=(output_dir, limit_per_file, burst_size, block_size, file_name_list, file_name_index, results, id_source), )
        p.start()
        p.join()
        mem = psutil.virtual_memory()[2]
        memory_history.append(mem)
        if p.exitcode != 0:
            print(f"Farmer stopped with exit code {p.exitcode}, load {psutil.getloadavg()[2] / os.cpu_count()}, "
                  f"mem {mem}%")
        p.close()

        try:
            result = results.get(block=True, timeout=5)
        except Empty:
            print("Result queue was empty for some reason. An error was likely raised inside the farmer.")
            break

        if result['status'] != "success":
            print(f"Farmer yielded result {result}")
            break

        bar.update(file_name_index.value - last_index_pos)
        last_index_pos = file_name_index.value
        
    bar.close()
    plt.plot(range(len(memory_history)), memory_history)
    plt.savefig("memory_history.png")