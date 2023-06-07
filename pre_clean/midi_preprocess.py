# midi_preprocess.py

import os
import sys
import json
import glob
import math
import mido as md
import shutil
import subprocess
import pretty_midi # pip install pretty_midi
import joblib # pip install joblib
import urllib.request

INST_TYPE_MAPPING = {
        (0, 8): 0,  # Piano
        (8, 16): 0,  # Chromatic Percussion
        (16, 24): 0,  # Organ
        (24, 32): 25,  # Guitar
        (32, 40): 43,  # Bass
        (40, 43): 48,  # String
        (43,44) : 43, # Double Bass
        (44,56) : 48, #String enesemble
        (56, 64): 48,  # Brass
        (64, 72): 48,  # Reed
        (72, 80): 48,  # Pipe
        (80, 88): 48,  # Synth Lead -> String
        (88, 96): 48,  # Synth Pad
        (96, 104): 48,  # Synth Effect
        (104, 112): 48,  # Ethnic
        (112, 120): 114,  # Percussive
        (120, 129): 114,  # Sound Effects
    }

def contains_inst(midi, program):
    for inst in midi.instruments:
        if inst.program == program:
            return True
    return False

def list_contains_inst(instruments, program):
    for inst in instruments:
        if inst.program == program:
            return True
    return False

def get_inst(midi, program):
    for inst in midi.instruments:
        if inst.program == program:
            return inst
    return None

def list_get_inst(instruments, program):
    for inst in instruments:
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

def list_get_inst_index(instruments, program):
    i=0
    for inst in instruments:
        if inst.program == program:
            return i
        else:
            i += 1
    return None

def inst_to_type(program):
    for (l, h) in INST_TYPE_MAPPING:
        if l <= program < h:
            return INST_TYPE_MAPPING[(l, h)]
    return None

def process_midi(midi_file, PROC_DATA_PATH, MELODY_OUTPUT_PATH):
    '''
    1. Processes input midi file
    2. Stores the processed file into PROC_DATA_PATH if processing was successful
    3. Returns True or False depending on the success of processing

    Input
    ----------
    midi : File path to the midi file

    Output
    ----------
    valid : True if input midi is successfully processed, False otherwise
    '''

    STORE_PATH = os.path.join(PROC_DATA_PATH, midi_file.split('/')[-1])
    MIDI_NAME = midi_file.split('/')[-1]
    
    # TODO: Process instruments and etc.
    try:
        midi = pretty_midi.PrettyMIDI(midi_file)
    except Exception as e:
        print(f'Passed {midi_file}')
        return False

    new_midi = pretty_midi.PrettyMIDI(midi_file)
    new_instruments = []

    ###############################################################################
    # 1. Melody Extraction

    melody_exists = True
    # Obtain melody
    with open(os.path.join(MELODY_OUTPUT_PATH,'program_result.json'), 'r') as f:
        programs = json.load(f)
    if midi_file in programs:
        melody_program = programs[midi_file]['melody']
    # If midi mider can't extract melody, set flute as the melody
    elif contains_inst(new_midi, 73):
        melody_program = 73
    # If flute doesn't exist, discard this midi
    else:
        #print(f'There is no melody in {midi_file}')
        melody_exists = False

    # Let square synthisizer(80) play the melody
    if melody_exists:
        melody_inst = get_inst(new_midi, melody_program)
        melody_inst_idx = get_inst_index(new_midi, melody_program)
        assert melody_inst != None, 'Melody extraction error'
        assert melody_inst_idx != None, 'Melody extraction error'
        melody_inst.program = 80
        melody_inst.name = pretty_midi.program_to_instrument_name(80)
        
        # Remove the previous melody instrument
        del new_midi.instruments[melody_inst_idx]
        syn_inst_idx = get_inst_index(new_midi, 80)
        if syn_inst_idx != None:
            del new_midi.instruments[syn_inst_idx]
        new_instruments.append(melody_inst)

    ###############################################################################
    # 2. Track Compression

    # Compress the piano-rolls of the five instruments
    for inst_type in [0, 25, 43, 48]:
        new_inst = pretty_midi.Instrument(inst_type, is_drum=False,
                                          name=pretty_midi.program_to_instrument_name(inst_type))
        new_instruments.append(new_inst)
    for inst_type in [114]:
        new_inst = pretty_midi.Instrument(inst_type, is_drum=True,
                                          name=pretty_midi.program_to_instrument_name(inst_type))
        new_instruments.append(new_inst)

    for inst in new_midi.instruments:
        inst_type = inst_to_type(inst.program)
        if inst_type != 43:
            new_inst = list_get_inst(new_instruments, inst_type)
            new_inst_idx = list_get_inst_index(new_instruments, inst_type)
            for note in inst.notes:
                new_inst.notes.append(note)
            del new_instruments[new_inst_idx]
            new_instruments.append(new_inst)

        # Choose the bass track with most notes
        else:
            new_inst = list_get_inst(new_instruments, inst_type)
            new_inst_idx = list_get_inst_index(new_instruments, inst_type)
            if len(new_inst.notes) < len(inst.notes):
                del new_instruments[new_inst_idx]
                new_inst = inst
                new_inst.program = inst_type
                new_instruments.append(new_inst)

    new_midi.instruments = new_instruments

    ###############################################################################
    # 3. Data Filtration: Optional
    # Discard midis that contain less than 2 above-20-note tracks


    ###############################################################################
    # 4. Data Segmentation
    # Keep only the 4/4 time signature segments for each midi: Optional

    # Or, discard midis that contain segments that are not 4/4 ts
    #if (len(new_midi.time_signature_changes) != 1 or
    #    new_midi.time_signature_changes[0].numerator != 4 or
    #    new_midi.time_signature_changes[0].denominator != 4):
    #    return False

    new_midi.write(STORE_PATH)

    ###############################################################################
    # 5. Do other things with mido - Optional

    return True