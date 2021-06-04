import torch.optim
import torch.utils.data
from dataset import *
from utils import *
import torch.nn.functional as F
from models import *
from tqdm import tqdm

def idx2string(idx, idx2word):
    words = []
    for word_idx in idx:
        words.append(idx2word[word_idx])
    return words

def pitch_match(pred_words, target_words):
    def get_pitch(word):
        if word[-1] == '6':
            return word[:-2]
        else:
            return word[:-1]

    pred_pitches, target_pitches = [], []
    for w in pred_words:
        if w[0].lower() not in 'abcdefgr':
            continue
        pred_pitches.append(get_pitch(w))
    for w in target_words:
        if w[0].lower() not in 'abcdefgr':
            continue
        target_pitches.append(get_pitch(w))
    matches = 0
    if len(pred_pitches) < len(target_pitches):
        for i in range(len(pred_pitches)):
            if pred_pitches[i] == target_pitches[i]:
                matches += 1
    else:
        for i in range(len(target_pitches)):
            if pred_pitches[i] == target_pitches[i]:
                matches += 1
    return matches / (max(len(pred_pitches), len(target_pitches)) + 0.0001)


def beat_match(pred_words, target_words):
    def get_beat(word):
        if word[-1] == '6':
            return word[-2:]
        else:
            return word[-1]

    pred_beats, target_beats = [], []
    for w in pred_words:
        if w[0].lower() not in 'abcdefgr':
            continue
        pred_beats.append(get_beat(w))
    for w in target_words:
        if w[0].lower() not in 'abcdefgr':
            continue
        target_beats.append(get_beat(w))
    matches = 0
    if len(pred_beats) < len(target_beats):
        for i in range(len(pred_beats)):
            if pred_beats[i] == target_beats[i]:
                matches += 1
    else:
        for i in range(len(target_beats)):
            if pred_beats[i] == target_beats[i]:
                matches += 1
                # clear divide by zero errors
    return matches / (max(len(pred_beats), len(target_beats)) + 0.0001)


if __name__ == '__main__':
    args = dict(label_type="word", emb_dim=20, decoder_dim=300, att_dim=300, dropout=0.5,
                batch_size=32,
                workers=0, encoder_lr=1e-4, decoder_lr=1e-4, decay_rate=1, grad_clip=5.0, att_reg=1.0,
                print_freq=100, save_freq=10,
                checkpoint=None, data_dir=None, label_file=None,
                model_name="base", beam_size=10, model_size=34)
    print("EM: " + str(EM))
    print("pitch: " + str(pitch))
    print("beat: " + str(beat))



