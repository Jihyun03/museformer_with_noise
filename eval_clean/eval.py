# eval.py

import os
import sys
import glob
import math
import shutil
import random
import subprocess
import pretty_midi # pip install pretty_midi
import joblib # pip install joblib

###############################################################################
# 0. Configuration variables

NUM_REPEAT = 3000
NUM_REFERENCE = 50
NUM_CLASS = 255
NGRAM = [2, 3, 4]
NGRAM_WEIGHT = {2: 1/3, 3: 1/3, 4: 1/3}

###############################################################################
# 1. Midi helper functions

def contains_inst(midi, program):
    for inst in midi.instruments:
        if inst.program == program:
            return True
    return False

def get_inst(midi, program):
    for inst in midi.instruments:
        if inst.program == program:
            return inst
    return None

def get_inst_index(midi, program):
    i=0
    for inst in midi.instruments:
        if inst.program == program:
            return i
        else:
            i += 1
    return None

###############################################################################
# 2. Midi to text

def pitch_diff_to_class(pitch_diff):
    global NUM_CLASS
    pitch_diff += 127
    num_per_class = [0] * NUM_CLASS
    for i in range(255):
        num_per_class[i % NUM_CLASS] += 1
    for i in range(NUM_CLASS):
        if pitch_diff < num_per_class[i]:
            return i
        pitch_diff -= num_per_class[i]
    assert False, 'Conversion from pitch difference to class failed'

def midi_to_txt(midi_path, TXT_PATH):
    MIDI_NAME = midi_path.split('/')[-1].split('.')[0]
    
    midi = pretty_midi.PrettyMIDI(midi_path)
    if not contains_inst(midi, 80):
        print(f'{midi_path} has no square synthesizer')
        return False
    
    f = open(os.path.join(TXT_PATH, MIDI_NAME + '.txt'), 'w')
    melody_inst = get_inst(midi, 80)
    pitch = None
    for note in melody_inst.notes:
        if pitch == None:
            pitch = note.pitch
        else:
            prev_pitch = pitch
            pitch = note.pitch
            f.write(str(pitch_diff_to_class(pitch - prev_pitch)))
            f.write('\n')
    f.close()

    return True

def midis_to_txt(MIDI_PATH, TXT_PATH):
    if os.path.isdir(TXT_PATH):
        print(f'Text files already generated into {TXT_PATH}')
        print('To overwrite, delete it and then retry')
    else:
        print(f'Start generating text files into {TXT_PATH}')
        os.makedirs(TXT_PATH, exist_ok=True)

        midi_list = glob.glob(os.path.join(MIDI_PATH, '*.mid'))[0:1000]
        statistics = joblib.Parallel(n_jobs=-1, verbose=1)(
            joblib.delayed(midi_to_txt)(midi, TXT_PATH)
            for midi in midi_list)
        statistics = [s for s in statistics if s is not None]
        stat = statistics.count(True)
        
        print(f'Finished generating files. {stat} files contained the melody.')
    print()

###############################################################################
# 3. Generate hash table

def hash(lenk_class_seq, ngram):
    global NUM_CLASS
    assert len(lenk_class_seq) == ngram, 'Interval class sequence length mismatch'
    
    index = 0
    for i in range(ngram):
        index += (NUM_CLASS ** i) * lenk_class_seq[i]
    return index

def generate_hash_table_one_txt(txt_path, HASH_PATH):
    global NGRAM
    TXT_NAME = txt_path.split('/')[-1].split('.')[0]

    f_txt = open(os.path.join(txt_path), 'r')
    class_seq = f_txt.read().splitlines()

    for ngram in NGRAM:
        NGRAM_PATH = os.path.join(HASH_PATH, str(ngram))
        f_hash = open(os.path.join(NGRAM_PATH, TXT_NAME + '.txt'), 'w')
        hash_table = {}
        for i in range(len(class_seq) - ngram + 1):
            lenk_class_seq = [int(i) for i in class_seq[i:i+ngram]]
            index = hash(lenk_class_seq, ngram)
            if index not in hash_table:
                hash_table[index] = 1
            else:
                hash_table[index] += 1
        for index in hash_table:
            f_hash.write(str(index) + ':' + str(hash_table[index]))
            f_hash.write('\n')
        f_hash.close()

    f_txt.close()   

def generate_hash_table(TXT_PATH, HASH_PATH):
    if os.path.isdir(HASH_PATH):
        print(f'Hash already generated into {HASH_PATH}')
        print('To overwrite, delete it and then retry')
    else:
        print(f'Start generating hash into {HASH_PATH}')
        os.makedirs(HASH_PATH, exist_ok=True)
        for ngram in NGRAM:
            NGRAM_PATH = os.path.join(HASH_PATH, str(ngram))
            os.makedirs(NGRAM_PATH, exist_ok=True)
        txt_list = glob.glob(os.path.join(TXT_PATH, '*.txt'))
        joblib.Parallel(n_jobs=-1, verbose=1)(
            joblib.delayed(generate_hash_table_one_txt)(txt, HASH_PATH)
            for txt in txt_list)
        
        print(f'Finished generating hash.')
    print()

###############################################################################
# 4. Compute self-bleu

def hash_file_to_dict(hash_file_path):
    f = open(hash_file_path, 'r')
    f_list = f.read().splitlines()
    dictionary = {}
    for s in f_list:
        key, value = s.split(':')
        dictionary[int(key)] = int(value)
    f.close()
    return dictionary

def compute_bp(hash_file_name, HASH_PATH):
    # Optional: Compute brevity penalty
    return 1

def compute_precision(cand_dict, ref_dict_list):
    sum_clipped_count = 0
    sum_unclipped_count = 0
    for index in cand_dict:
        max_occur = 0
        for ref_dict in ref_dict_list:
            if index in ref_dict:
                if ref_dict[index] > max_occur:
                    max_occur = ref_dict[index]
        if cand_dict[index] < max_occur:
            sum_clipped_count += cand_dict[index]
        else:
            sum_clipped_count += max_occur
        sum_unclipped_count += cand_dict[index]
    
    if sum_unclipped_count < 0 or sum_clipped_count < 0:
        assert False, 'Found negative count'
    elif sum_unclipped_count == 0:
        return None
    elif sum_clipped_count == 0:
        return None
    else:
        return sum_clipped_count / sum_unclipped_count

def compute_bleu(hash_file_name, full_dict, HASH_PATH):
    global NGRAM_WEIGHT, NGRAM
    assert len(NGRAM_WEIGHT) == len(NGRAM), 'n-gram weight length mismatch'

    sum_wlogp = 0
    for ngram in NGRAM:
        cand_dict = full_dict[ngram][hash_file_name]
        ref_dict_list = []
        for hash in full_dict[ngram]:
            if hash == hash_file_name:
                continue
            else:
                ref_dict_list.append(full_dict[ngram][hash])

        w = NGRAM_WEIGHT[ngram]
        p = compute_precision(cand_dict, ref_dict_list)
        if p == None:
            return None
        else:
            sum_wlogp += w * math.log(p)
    
    bp = compute_bp(hash_file_name, HASH_PATH)
    bleu = bp * math.exp(sum_wlogp)

    return bleu

def compute_self_bleu(TXT_PATH, HASH_PATH):
    hash_list = glob.glob(os.path.join(TXT_PATH, '*.txt'))
    random.shuffle(hash_list)
    hash_list = hash_list[0:NUM_REFERENCE]
    hash_name_list = [hash.split('/')[-1] for hash in hash_list]

    #print('Start retrieving dictionaries')
    full_dict = {}
    for ngram in NGRAM:
        NGRAM_PATH = os.path.join(HASH_PATH, str(ngram))
        dictionary = {}
        for hash_file_name in hash_name_list:
            HASH_FILE_PATH = os.path.join(NGRAM_PATH, hash_file_name)
            dictionary[hash_file_name] = hash_file_to_dict(HASH_FILE_PATH)
        full_dict[ngram] = dictionary
    #print('Finished retrieving dictionaries                                ')
    #print()

    #print('Start computing self-bleu')
    bleu_list = joblib.Parallel(n_jobs=-1, verbose=0)(
            joblib.delayed(compute_bleu)(hash_file_name, full_dict, HASH_PATH)
            for hash_file_name in hash_name_list)
    bleu_list = [bleu for bleu in bleu_list if bleu is not None]
    sum_bleu = sum(bleu_list)
    self_bleu = sum_bleu / len(hash_name_list)
    #print('Finished computing self-bleu          ')
    #print(f'self_bleu = {self_bleu:.3f}')
    return self_bleu

###############################################################################

def main():
    # Path constants
    MY_PATH = os.path.realpath(os.path.dirname(__file__))
    MIDI_PATH = os.path.join(MY_PATH, 'midi')
    TXT_PATH = os.path.join(MY_PATH, 'txt')
    HASH_PATH = os.path.join(MY_PATH, 'hash')

    midis_to_txt(MIDI_PATH, TXT_PATH)
    generate_hash_table(TXT_PATH, HASH_PATH)
    self_bleu_sum = 0
    for i in range(NUM_REPEAT):
     self_bleu_sum += compute_self_bleu(TXT_PATH, HASH_PATH)
     print(f'{i}/{NUM_REPEAT}\r', end='')
    print(f'Mean self_bleu: {self_bleu_sum / NUM_REPEAT}')
    return 0

if __name__ == '__main__':
    sys.exit(main())