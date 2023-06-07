# data_preprocess.py
# 1. Downloads LMD-full
# 2. Processes the dataset into museformer compatible, compact form (4/4, 6 instruments)
# 3. Writes MIDI file list {test, train, valid}.txt and copies MIDI files into data/midi

import os
import sys
import glob
import math
import shutil
import subprocess
import pretty_midi # pip install pretty_midi
import joblib # pip install joblib
import urllib.request

import midi_preprocess

###############################################################################
# 0. Configuration variables

# If true, copy midi files into data/midi - Use with museformer
COPY_MIDI = True

###############################################################################
# 1. Obtaining dataset

def show_progress(block_num, block_size, total_size):
    downloaded_mb = block_num * block_size / 1000000
    total_size_mb = total_size / 1000000
    print(f'  {downloaded_mb:.2f}MB / {total_size_mb:.2f}MB\r', end='')

def download_data(dl_files):
    for dl_name, dl in dl_files.items():
        if os.path.isfile(dl["path"]):
            print(f'{dl_name} already downloaded to {dl["path"]}')
            print('To re-download, delete it and then retry')
        else:
            print(f'Start downloading {dl_name} to {dl["path"]}')
            urllib.request.urlretrieve(dl["url"], dl["path"], show_progress)
            print(f'\rFinished downloading {dl_name}')
        print()

def extract_data(MY_PATH, ARCH_DATA_PATH, dl_files):
    if os.path.isdir(ARCH_DATA_PATH):
        print(f'{dl_files["midi"]["path"]} already extracted to {ARCH_DATA_PATH}')
        print('To re-extract, delete it and then retry')
    else:
        print('Start extracting archive')
        subprocess.check_output([
            'tar', '-xzf', dl_files["midi"]["path"],
            '-C', MY_PATH
        ])
        print('Finished extracting')
    print()

###############################################################################
# 2. Processing dataset

def get_melody(MELODY_INPUT_PATH, MELODY_OUTPUT_PATH, DEFAULT_META_PATH, ARCH_DATA_PATH):
    if os.path.isdir(MELODY_OUTPUT_PATH) or os.path.isdir(MELODY_INPUT_PATH):
        print(f'Melodies already obtained from {MELODY_INPUT_PATH} into {MELODY_OUTPUT_PATH}')
        print('To overwrite, delete them and then retry')
    else:
        print(f'Start copying midi files into {MELODY_INPUT_PATH}')
        os.makedirs(MELODY_INPUT_PATH, exist_ok=True)
        midi_list = []
        for name in ['test.txt', 'train.txt', 'valid.txt']:
            f = open(os.path.join(DEFAULT_META_PATH, name), 'r')
            for line in f.readlines():
                line = line.strip()
                midi_list.append(os.path.join(ARCH_DATA_PATH, line[0], line))
        i=1
        for midi in midi_list:
            print(f'  {i} / {len(midi_list)}\r', end='')
            shutil.copy(midi, MELODY_INPUT_PATH)
            i += 1
        print('Finished copying midi files           \n')

        print(f'Obtaining melodies into {MELODY_OUTPUT_PATH}')
        os.makedirs(MELODY_OUTPUT_PATH, exist_ok=True)
        subprocess.check_output([
            'python3', './midi-miner/track_separate.py',
            '-i', MELODY_INPUT_PATH,
            '-o', MELODY_OUTPUT_PATH
        ])
        print('Finished obtaining melodies')
    print()

def process_default_data(PROC_DATA_PATH, DEFAULT_META_PATH, MELODY_INPUT_PATH, MELODY_OUTPUT_PATH):
    if os.path.isdir(PROC_DATA_PATH):
        print(f'Midi files already processed into {PROC_DATA_PATH}')
        print('To overwrite, delete it and then retry')
    else:
        print(f'Start processing default midi files into {PROC_DATA_PATH}')
        os.makedirs(PROC_DATA_PATH, exist_ok=True)

        midi_list = []
        for name in ['test.txt', 'train.txt', 'valid.txt']:
            f = open(os.path.join(DEFAULT_META_PATH, name), 'r')
            for line in f.readlines():
                line = line.strip()
                midi_list.append(os.path.join(MELODY_INPUT_PATH, line))

        statistics = joblib.Parallel(n_jobs=-1, verbose=1)(
            joblib.delayed(midi_preprocess.process_midi)(midi, PROC_DATA_PATH, MELODY_OUTPUT_PATH)
            for midi in midi_list)
        statistics = [s for s in statistics if s is not None]
        stat = statistics.count(True)
        print(f'Finished processing files. {stat} files were valid out of {len(midi_list)}')
    print()

###############################################################################
# 3. Writing file list and midi

def write_filelist(PROC_DATA_PATH, PROC_META_PATH):
    # Write MIDI file list into PROC_META_PATH
    if os.path.isdir(PROC_META_PATH):
        print(f'Midi file list already written into {PROC_META_PATH}')
        print('To overwrite, delete it and then retry')
    else:
        print(f'Writing midi file list into {PROC_META_PATH}')
        os.makedirs(PROC_META_PATH, exist_ok=True)

        midi_list_all = glob.glob(os.path.join(PROC_DATA_PATH, '*.mid'))
        NUM_TEST = math.ceil(len(midi_list_all) / 10)
        NUM_TRAIN = math.ceil(len(midi_list_all) * 8 / 10)
        NUM_VALID = len(midi_list_all) - NUM_TEST - NUM_TRAIN
        file_list = {"test.txt" : midi_list_all[0:NUM_TEST],
                     "train.txt" : midi_list_all[NUM_TEST:NUM_TEST + NUM_TRAIN],
                     "valid.txt" : midi_list_all[NUM_TEST + NUM_TRAIN:len(midi_list_all)]}
        for name in file_list:
            midi_list = file_list[name]
            f = open(os.path.join(PROC_META_PATH, name), 'a')
            for i in range(len(midi_list)):
                f.write(midi_list[i].split('/')[-1] + '\n')
            f.close()

        print('Finished writing file list')
    print()

def copy_midi(PROC_DATA_PATH):
    COPY_PATH = os.path.join(PROC_DATA_PATH, '../../data/midi')
    if os.path.isdir(COPY_PATH):
        print(f'Midi files already copied into {COPY_PATH}')
        print('To overwrite, delete it and then retry')
    else:
        print(f'Copying midi files into {COPY_PATH}')
        os.makedirs(COPY_PATH, exist_ok=True)
        midi_list_all = glob.glob(os.path.join(PROC_DATA_PATH, '*.mid'))
        i=1
        for midi in midi_list_all:
            print(f'  {i} / {len(midi_list_all)}\r', end='')
            shutil.copy(midi, COPY_PATH)
            i += 1
        print('Finished copying midi files           \n')

###############################################################################

def main():
    # Path constants
    MY_PATH = os.path.realpath(os.path.dirname(__file__))
    ARCH_DATA_PATH = os.path.join(MY_PATH, 'lmd_full')
    PROC_DATA_PATH = os.path.join(MY_PATH, 'processed_data')
    DEFAULT_META_PATH = os.path.join(MY_PATH, 'default_meta')
    PROC_META_PATH = os.path.join(MY_PATH, 'processed_meta')
    MELODY_INPUT_PATH = os.path.join(MY_PATH, 'midi-miner/example/input')
    MELODY_OUTPUT_PATH = os.path.join(MY_PATH, 'midi-miner/example/output')

    # Files to be downloaded
    dl_files = {
        "midi" : {
            "path" : os.path.join(MY_PATH, 'lmd_full.tar.gz'),
            "url" : 'http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz'
        },
        "json" : {
            "path" : os.path.join(MY_PATH, 'md5_to_paths.json'),
            "url" : 'http://hog.ee.columbia.edu/craffel/lmd/md5_to_paths.json'
        }
    }

    download_data(dl_files)
    extract_data(MY_PATH, ARCH_DATA_PATH, dl_files)
    get_melody(MELODY_INPUT_PATH, MELODY_OUTPUT_PATH, DEFAULT_META_PATH, ARCH_DATA_PATH)
    process_default_data(PROC_DATA_PATH, DEFAULT_META_PATH, MELODY_INPUT_PATH, MELODY_OUTPUT_PATH)
    write_filelist(PROC_DATA_PATH, PROC_META_PATH)
    if COPY_MIDI:
        copy_midi(PROC_DATA_PATH)

    return 0

if __name__ == '__main__':
    sys.exit(main())