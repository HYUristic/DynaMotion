import mido
from mido import MidiFile
import numpy as np


def update_piano_roll(message: mido.Message,
                      piano_roll: np.ndarray,
                      velocity: np.ndarray,
                      state: dict):
    # Update to current tick
    for delta_tick in range(message.time):
        cur_tick = state['tick'] + delta_tick
        if state['sustain'] and state['tick'] != 0:  # pedal PRESSED
            previous_note = piano_roll[state['tick']-1]
            concat_note = np.where((previous_note + state['pressed']) > 0, 1, 0)
            piano_roll[cur_tick] = concat_note
        else:  # pedal NOT PRESSED
            piano_roll[cur_tick] = state['pressed']

    state['tick'] += message.time

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


def merge(new_index, original_index, new, old):
    new[new_index] += old[original_index] * np.where(new[new_index] > 0, 0, 1)


def resize(piano_roll, velocity, desired_number_of_ticks):
    new_piano_roll = np.zeros((desired_number_of_ticks, 128)).astype(int)
    new_velocity = np.zeros((desired_number_of_ticks, 128)).astype(float)

    original_number_of_ticks = piano_roll.shape[0]
    difference_ratio = desired_number_of_ticks / original_number_of_ticks

    for original_index in range(original_number_of_ticks):
        new_index = int(difference_ratio * original_index)

        merge(new_index, original_index, new_piano_roll, piano_roll)
        merge(new_index, original_index, new_velocity, velocity)

    return new_piano_roll, new_velocity


def midi_to_numpy(midi_path: str, quantization_period: float):
    print('start midi')
    midi_file = MidiFile(midi_path)

    # Configure midi settings
    ticks_per_beat = midi_file.ticks_per_beat
    for message in midi_file.tracks[0]:
        if message.type == 'set_tempo':
            microsec_per_beat = message.tempo
    microsec_per_ticks = microsec_per_beat / ticks_per_beat
    sec_per_ticks = microsec_per_ticks * 1e-6
    length_sec = midi_file.length
    length_tick = int(length_sec / sec_per_ticks)

    track = midi_file.tracks[1]
    piano_roll = np.zeros((length_tick, 128)).astype(int)
    velocity = np.zeros((length_tick, 128)).astype(float)
    state = {
        'tick': 0,
        'pressed': np.zeros(128).astype(int),
        'sustain': False
    }

    for message in track:
        update_piano_roll(message=message, piano_roll=piano_roll, velocity=velocity, state=state)
    print('finished')
    # Resize
    desired_number_of_ticks = int((length_tick / ticks_per_beat) / quantization_period)
    piano_roll, velocity = resize(piano_roll, velocity, desired_number_of_ticks)

    return piano_roll, velocity


if __name__ == '__main__':
    test_path = '/Users/litcoderr/Downloads/MIDI (1)/midi.mid'
    numpy = midi_to_numpy(test_path, 1/8)