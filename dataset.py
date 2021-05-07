import torch
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
import itertools


class Dataset(torch.utils.data.Dataset):

	def __init__(self, list_IDs, labels):
		'Initialization'
		self.labels = labels
		self.list_IDs = list_IDs

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def __getitem__(self, index):
		ID = self.list_IDs[index]
		img = Image.open("data/image" + str(ID) + "-1.png")
		resizer = transforms.Resize((256, 256))
		img = resizer(img)
		# repeat the grey scale image along the channel dimension
		X = torch.tensor(np.repeat(np.array(img)[:, :, 3][np.newaxis, :, :], 3, 0))
		y = torch.tensor(self.labels[ID][0])
		cap_len = torch.tensor([self.labels[ID][1]])
		return X, y, cap_len


def read_captions(file):
	"""
	file: file path
	return: a list of captions, word2idx mapping, max_length
	"""
	with open(file, "r") as fb:
		lines = fb.readlines()
	lines = [list(line.lstrip("tinyNotation: ").rstrip(" \n")) for line in lines]

	# naive implementation of char list. can do more sophisticated such as ## , --.
	chars = ["C", "D", "E", "F", "G", "A", "B", "c", "d", "e", "f", "g", "a", "b", "r"]
	nums = [str(num) for num in list(range(0, 9))]
	special = [" ", "'", "-", "#", "/", "<start>", "<end>", "<pad>"]
	char_list = chars + nums + special
	word2idx = {char: idx for idx, char in enumerate(char_list)}

	max_len = find_max_len(lines)

	return lines, word2idx, max_len

def read_captions_word(file):
	"""
	file: file path
	return: a list of captions, word2idx mapping, max_length
	"""
	with open(file, "r") as fb:
		lines = fb.readlines()
	lines = [line.lstrip("tinyNotation: ").rstrip().split(" ") for line in lines]

	pitches_nats = ["C", "D", "E", "F", "G", "A", "B",
				"c", "d", "e", "f", "g", "a", "b",
				"c'", "d'", "e'", "f'", "g'", "a'", "b'", 
				"c''", "d''", "e''", "f''", "g''", "a''", "b''", 
				"r", "r"]
	accidentals = ["", "#", "#", "##", "-", "-", "--"]
	beats = [str(num) for num in [1,2,4,8,16]]

	notes = itertools.product(pitches_nats, accidentals, beats)
	notes = [note[0]+note[1]+note[2] for note in notes]
	time_sigs = ["4/4", "2/4", "3/4", "3/8", "6/8", "3/2"]
	special = ["<start>", "<end>", "<pad>"]
	word_list = time_sigs + notes + special
	word2idx = {word: idx for idx, word in enumerate(word_list)}

	max_len = find_max_len(lines)

	return lines, word2idx, max_len


# need to pad each caption to the maximum length of the caption in the dataset
def find_max_len(captions):
	"""
	:param captions: a list of captions (each caption is organized as a list)
	:return: max length
	"""
	max_len = 0
	for line in captions:
		if len(line) > max_len:
			max_len = len(line)
	return max_len


def convert_sentence_idx(word2idx, sentence, max_len):
	"""
	true max_len = max_len + 2 because of <start> and <end>
	:param word2idx: a word to index map
	:param sentence: one sentence of words
	:param max_len: sentence max length in the corpus
	:return: a list of indices by looking up word2idx
	"""

	max_len = max_len + 2
	idx_list = [word2idx["<start>"]]
	idx_list = idx_list + [word2idx[str(word)] for word in sentence]
	idx_list.append(word2idx["<end>"])
	original_len = len(idx_list)

	if original_len < max_len:
		idx_list = idx_list + [word2idx["<pad>"]] * (max_len - original_len)

	return torch.tensor(idx_list, dtype=int), original_len

def convert_corpus_idx(word2idx, corpus, max_len):
	"""
	A wrapper for convert sentence idx to convert all sentences in the corpus to lists of indices
	:param word2idx: word2idx map
	:param corpus: corpus consisting of lists of sentences
	:param max_len: max length of sentence
	:return: a list of lists of indicies
	"""
	corpus_idx = []
	for line in corpus:
		line_idx, cap_len = convert_sentence_idx(word2idx, line, max_len)
		corpus_idx.append((line_idx, cap_len))

	return corpus_idx

if __name__ == '__main__':
	corpus, word2idx, max_len = read_captions("music_strings_small.txt")
	corpus_idx = convert_corpus_idx(word2idx, corpus, max_len)
	dataset = Dataset(list(range(0, len(corpus_idx))), corpus_idx)
	print(dataset.__len__())
	X, y, cap_len = dataset.__getitem__(1)
	print(X.shape)
	print(y.shape)
	print(cap_len)
