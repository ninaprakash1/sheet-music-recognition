import torch
import torch.nn as nn
from dataset import *
import torch.utils.data

class ConvNet(nn.Module):
	def __init__(self, in_channel, channel_size, kernel=5, pad=2, num_conv=2, embed=20, output_width=400):
		super().__init__()

		self.conv1 = nn.Conv2d(in_channel, channel_size, kernel, padding=pad)
		module = []
		module.append(self.conv1)
		for i in range(num_conv - 1):
			module.append(nn.Conv2d(channel_size, channel_size, kernel, padding=pad))
		for i in range(num_conv):
			nn.init.kaiming_normal_(module[i].weight)
			nn.init.zeros_(module[i].bias)

		self.sequence = nn.Sequential()
		self.sequence.add_module("conv" + str(0), module[0])
		self.sequence.add_module("relu" + str(0), nn.ReLU())
		self.sequence.add_module("conv" + str(1), module[1])
		self.sequence.add_module("relu" + str(1 + 0), nn.ReLU())
		self.sequence.add_module("pool1", nn.MaxPool2d((1, 2)))
		self.sequence.add_module("conv2", nn.Conv2d(1, 1, kernel_size=5, padding=2))
		self.sequence.add_module("relu2", nn.ReLU())
		self.sequence.add_module("conv3", nn.Conv2d(1, 1, kernel_size=5, padding=2))
		self.sequence.add_module("relu3", nn.ReLU())
		self.sequence.add_module("pool" + str(0), nn.AdaptiveMaxPool2d((embed, output_width)))
		# self.sequence.add_module("pool" + str(0), nn.MaxPool2d(2, padding=1))


	def forward(self, x):

		out = self.sequence(x)

		return out


if __name__ == '__main__':
	corpus, word2idx, idx2word, max_len = read_captions_word("music_strings_small.txt")
	corpus_idx = convert_corpus_idx(word2idx, corpus, max_len)
	data = Dataset("data", list(range(0, len(corpus))), corpus_idx)
	train_loader = torch.utils.data.DataLoader(
		data, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
	encoder = ConvNet(3, 1)
	for i, (imgs, caps, caplens) in enumerate(train_loader):
		imgs = encoder(imgs.float())
		print(imgs.shape)