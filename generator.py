import random
from music21 import *

# Note: Durations are *reversed* i.e., 1 represents a 16th-note
# 2 represents an 8th-note, 4 represents a quarter note, etc.

durations = [1, 2, 4, 8, 16]

switch = {1: 16,
	  2: 8,
	  4: 4,
	  8: 2,
	  16: 1}

bottom_map = {2: 8,
	      4: 4,
	      8: 2}

pitches_nats = ["C", "D", "E", "F", "G", "A", "B",
		"c", "d", "e", "f", "g", "a", "b",
		"c'", "d'", "e'", "f'", "g'", "a'", "b'", 
		"c''", "d''", "e''", "f''", "g''", "a''", "b''", 
		"r", "r"]
accidentals = ["", "", "", "", "", "", "", "#", "#", "##", "-", "-", "--"]

def generate_music_string(time_sig, num_measures=1):
	result = "tinyNotation: " + time_sig + " "
	time_sig = time_sig.split("/")
	beats = []
	top = int(time_sig[0])
	bottom = int(time_sig[1])
	while sum(beats) != top * bottom_map[bottom] * num_measures:
		new_beat = random.sample(durations, 1)[0]
		if sum(beats) + new_beat > top * bottom_map[bottom] * num_measures:
			continue
		else:
			beats.append(new_beat)
	beats = [switch[x] for x in beats]
	for beat in beats:
		pitch = random.sample(pitches_nats, 1)[0]
		acc = random.sample(accidentals, 1)[0]
		result += pitch
		if pitch != "r":
			result += acc
		result += str(beat)
		result += " "
	result_split = result[:-1].split(' ')[:34]
	return ' '.join(result_split)


