import time
import mido
from mido import MidiFile
import numpy as np
import matplotlib.pyplot as plt


def update_piano_roll(message: mido.Message,
                      piano_roll: np.ndarray,
                      velocity: np.ndarray,
                      state: dict,
                      rate_tick: float):

    # Update Piano Roll / State 
    tick_length = int(message.time * rate_tick)
    remainder = message.time - tick_length/rate_tick
    if remainder >= 0:
        state['remainder'] += remainder
    remainder_length = int(state['remainder'] * rate_tick)
    tick_length += remainder_length
    state['remainder'] -= remainder_length/rate_tick

    for delta_tick in range(tick_length):
        cur_tick = state['tick'] + delta_tick
        if state['sustain'] and state['tick'] != 0:  # pedal PRESSED
            previous_note = piano_roll[state['tick']-1]
            concat_note = np.where((previous_note + state['pressed']) > 0, 1, 0)
            piano_roll[cur_tick] = concat_note
        else:  # pedal NOT PRESSED
            piano_roll[cur_tick] = state['pressed']

    state['tick'] += tick_length

    # update state based on midi message
    if(message.type=='note_on'):
        if message.velocity == 0:
            state['pressed'][message.note] = 0
        else:
            state['pressed'][message.note] = 1
            velocity[state['tick']][message.note] = message.velocity / 127
    elif(message.type=='control_change'):
        if(message.control==64):
            if(message.value>=64):  # Pedal ON
                state['sustain'] = True
            else:  # Pedal OFF
                state['sustain'] = False


def midi_to_numpy(midi_path: str, quantization_period: float):
    midi_file = MidiFile(midi_path)

    # Configure midi settings
    # Original Midi settings (not quantized)
    ticks_per_beat = midi_file.ticks_per_beat
    for message in midi_file.tracks[0]:
        if message.type == 'set_tempo':
            microsec_per_beat = message.tempo
    microsec_per_ticks = microsec_per_beat / ticks_per_beat
    sec_per_ticks = microsec_per_ticks * 1e-6
    length_sec = midi_file.length
    length_tick = int(length_sec / sec_per_ticks)

    # New Midi settings (quantized tempo and length)
    new_length_tick = int((length_tick / ticks_per_beat) * quantization_period)
    rate_tick = new_length_tick / length_tick # skip length in ticks

    track = midi_file.tracks[1]
    piano_roll = np.zeros((new_length_tick, 128)).astype(bool)
    velocity = np.zeros((new_length_tick, 128)).astype(float)
    state = {
        'tick': 0,
        'remainder': 0,
        'pressed': np.zeros(128).astype(int),
        'sustain': False
    }
    for message in track:
        update_piano_roll(message=message, piano_roll=piano_roll, velocity=velocity, state=state, rate_tick=rate_tick)
    return piano_roll, velocity

