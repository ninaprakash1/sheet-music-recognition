"""
Adapted from problem set 3.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import copy
import sys
from models import SqueezeNet, ResNet


class ImageEncoder(nn.Module):
    """
    Transformer Encoder
    """
    def __init__(self, backbone="squeezenet", wordvec_dim=200, num_heads=4,
                 num_layers=2):

        super().__init__()
        if backbone == "squeezenet":
            self.backbone = SqueezeNet(emb_dim=wordvec_dim)
        elif backbone == "resnet18":
            self.backbone = ResNet(model_size=18, embed_dim=wordvec_dim)
        elif backbone == "resnet34":
            self.backbone = ResNet(model_size=34, embed_dim=wordvec_dim)
        self.num_heads = num_heads
        self.positional_encoding = PositionalEncoding(wordvec_dim)
        encoder_layer = TransformerEncoderLayer(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            if module.bias:
                module.bias.data.zero_()
            nn.init.xavier_uniform_(module.weight.data)

    def forward(self, features):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.

        Inputs:
         - features: image features, of shape (N, D, T') , where T' is the width of the image
         - captions: ground truth captions, of shape (N, T)

        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        x = self.backbone(features)
        x = self.positional_encoding(x)
        x = self.transformer(x)

        return x





class CaptioningTransformer(nn.Module):
    """
    A CaptioningTransformer produces captions from image features using a
    Transformer decoder.

    The Transformer receives input vectors of size D, has a vocab size of V,
    works on sequences of length T, uses word vectors of dimension W, and
    operates on minibatches of size N.
    """
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=2, max_length=50):
        """
        Construct a new CaptioningTransformer instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - num_heads: Number of attention heads.
        - num_layers: Number of transformer layers.
        - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self._null = word_to_idx["<pad>"]
        self._start = word_to_idx.get("<start>", None)
        self._end = word_to_idx.get("<end>", None)
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.num_heads = num_heads
        # TODO change the encoded image size
        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim)

        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)

        self.output = nn.Linear(wordvec_dim, vocab_size)

    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.

        Inputs:
         - features: image features, of shape (N, D, T') , where T' is the width of the image
         - captions: ground truth captions, of shape (N, T)

        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        N, T = captions.shape

        # Embed the captions.
        # shape: [N, T] -> [N, T, W]
        caption_embeddings = self.embedding(captions)
        caption_embeddings = self.positional_encoding(caption_embeddings)

        # transpose the image and transform feature representation of each vertical stripe
        # projected_features = self.visual_projection(features)
        # projected_features = F.relu(self.visual_projection(features.transpose(-1, -2)))
        projected_features = self.positional_encoding(features)

        # An additive mask for masking the future (one direction).
        # shape: [T, T]
        tgt_mask = torch.tril(torch.ones(T, T,
                                         device=caption_embeddings.device,
                                         dtype=caption_embeddings.dtype))

        # mask padded positions
        pad_mask = captions != self._null
        pad_mask = pad_mask.unsqueeze(1)
        tgt_mask = tgt_mask * pad_mask
        tgt_mask = tgt_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # Apply the Transformer decoder to the caption, allowing it to also
        # attend to image features.
        features = self.transformer(tgt=caption_embeddings,
                                    memory=projected_features,
                                    tgt_mask=tgt_mask)

        # Project to scores per token.
        # shape: [N, T, W] -> [N, T, V], V vocab size
        scores = self.output(features)

        return scores

    def sample(self, features, device, max_length=50, beam_size=10):
        """
        Given image features, use greedy decoding to predict the image caption.

        Inputs:
         - features: image features, of shape (1, D, T')
         - max_length: maximum possible caption length

        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        # with torch.no_grad():
        #     features = torch.Tensor(features)
        #     N = features.shape[0]
        #
        #     # Create an empty captions tensor (where all tokens are NULL).
        #     captions = self._null * np.ones((N, max_length), dtype=np.int32)
        #
        #     # Create a partial caption, with only the start token.
        #     partial_caption = self._start * np.ones(N, dtype=np.int32)
        #     partial_caption = torch.LongTensor(partial_caption)
        #     # [N] -> [N, 1]
        #     partial_caption = partial_caption.unsqueeze(1)
        #
        #     for t in range(max_length):
        #         # Predict the next token (ignoring all other time steps).
        #         output_logits = self.forward(features, partial_caption)
        #         output_logits = output_logits[:, -1, :]
        #
        #         # Choose the most likely word ID from the vocabulary.
        #         # [N, V] -> [N]
        #         word = torch.argmax(output_logits, dim=1)
        #
        #         # Update our overall caption and our current partial caption.
        #         captions[:, t] = word.numpy()
        #         word = word.unsqueeze(1)
        #         partial_caption = torch.cat([partial_caption, word], dim=1)
        #
        #     return captions


        with torch.no_grad():
            # initialze k the number of sequence we are decoding at each time step
            k = beam_size

            # Create an empty captions tensor (where all tokens are NULL).
            # captions = self._null * np.ones((N, max_length), dtype=np.int32)

            k_prev_words = torch.LongTensor([[self._start]] * k).to(device) # (k, 1)

            # Tensor to store top k sequences
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores
            top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()

            # Create a partial caption, with only the start token.
            # partial_caption = self._start * np.ones(N, dtype=np.int32)
            # partial_caption = torch.Tensor(partial_caption)
            # [N] -> [N, 1]
            # partial_caption = partial_caption.unsqueeze(1)


            # start with the token after <start>
            for step in range(1, max_length+1):

                # features = features.repeat(k, 1, 1)
                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, seqs)
                output_logits = output_logits[:, -1, :] # [k, V]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                # word = torch.argmax(output_logits, dim=1)

                scores = F.log_softmax(output_logits, dim=1)
                scores = top_k_scores.expand_as(scores) + scores

                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                prev_word_inds = top_k_words // self.vocab_size  # (s)
                next_word_inds = top_k_words % self.vocab_size  # (s)

                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != self._end]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)

                if k == 0:
                    break

                seqs = seqs[incomplete_inds]

                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                # k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if len(complete_seqs) > 0:
                i = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[i]
            else:
                i = int(scores.argmax() // scores.shape[1])
                seq = seqs[i].tolist()



                # Update our overall caption and our current partial caption.
                # captions[:, t] = word.numpy()
                # word = word.unsqueeze(1)
                # partial_caption = torch.cat([partial_caption, word], dim=1)

            return seq


class TransformerDecoderLayer(nn.Module):
    """
    A single layer of a Transformer decoder, to be used with TransformerDecoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=200, dropout=0.1):
        """
        Construct a TransformerDecoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.multihead_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()


    def forward(self, tgt, memory, tgt_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer.

        Inputs:
        - tgt: the sequence to the decoder layer, of shape (N, T, W)
        - memory: the sequence from the last layer of the encoder, of shape (N, S, D)
        - tgt_mask: the parts of the target sequence to mask, of shape (T, T)

        Returns:
        - out: the Transformer features, of shape (N, T, W)
        """
        # Perform self-attention on the target sequence (along with dropout and
        # layer norm).
        tgt2 = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Attend to both the target sequence and the sequence from the last
        # encoder layer.
        tgt2 = self.multihead_attn(query=tgt, key=memory, value=memory)
        # tgt = torch.cat([tgt, self.dropout2(tgt2)], dim=-1)
        # tgt = self.att_proj(tgt)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Pass
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer decoder, to be used with TransformerDecoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=200, dropout=0.1):
        """
        Construct a TransformerDecoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.multihead_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()


    def forward(self, features):
        """
        Pass the inputs (and mask) through the decoder layer.

        Inputs:
        - tgt: the sequence to the decoder layer, of shape (N, T, W)
        - memory: the sequence from the last layer of the encoder, of shape (N, S, D)
        - tgt_mask: the parts of the target sequence to mask, of shape (T, T)

        Returns:
        - out: the Transformer features, of shape (N, T, W)
        """
        # Perform self-attention on the target sequence (along with dropout and
        # layer norm).
        # Attend to both the target sequence and the sequence from the last
        # encoder layer.
        tgt = features
        tgt2 = self.multihead_attn(query=features, key=features, value=features)
        # tgt = torch.cat([tgt, self.dropout2(tgt2)], dim=-1)
        # tgt = self.att_proj(tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Pass
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt





def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)

        return output

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt):
        output = tgt

        for mod in self.layers:
            output = mod(output)

        return output


# class PositionalEncoding(nn.Module):
#     """
#     Encodes information about the positions of the tokens in the sequence. In
#     this case, the layer has no learnable parameters, since it is a simple
#     function of sines and cosines.
#     """
#
#     def __init__(self, embed_dim, dropout=0.1, max_len=1000):
#         """
#         Construct the PositionalEncoding layer.
#
#         Inputs:
#          - embed_dim: the size of the embed dimension
#          - dropout: the dropout value
#          - max_len: the maximum possible length of the incoming sequence
#         """
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         assert embed_dim % 2 == 0
#
#         pe = torch.zeros(1, max_len, embed_dim)
#
#         even = torch.arange(0, embed_dim, 2)
#         even_mat = torch.tensor(list(range(0, max_len))).unsqueeze(1).repeat(1, len(even))
#         coef = torch.exp(math.log(10000) * -1 * even / embed_dim)
#         even_mat = even_mat * coef
#
#         pe[0, :, 0::2] = torch.sin(even_mat)
#         pe[0, :, 1::2] = torch.cos(even_mat)
#
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         """
#         Element-wise add positional embeddings to the input sequence.
#
#         Inputs:
#          - x: the sequence fed to the positional encoder model, of shape
#               (N, S, D), where N is the batch size, S is the sequence length and
#               D is embed dim
#         Returns:
#          - output: the input sequence + positional encodings, of shape (N, S, D)
#         """
#         N, S, D = x.shape
#         sys.stdout.write(str(S) + "\t")
#         sys.stdout.write(str(x.shape) + "\t")
#         sys.stdout.write(str(self.pe.shape) + "\t")
#         sys.stdout.flush()
#         output = x + self.pe[:, :S, :]
#         output = self.dropout(output)
#
#         return output
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #sys.stdout.write(str(x.shape))
        #sys.stdout.write(str(self.pe.shape))
        #sys.stdout.flush()
        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)




class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.A = nn.Linear(embed_dim, embed_dim)


    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (T, S) where mask[i,j] == 0 indicates a token
         i in the target should not be influenced by token j in the source.

        Returns:
        - output: Tensor of shape (N, T, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        Nq, S, D = query.shape
        Nv, T, D = value.shape

        q = self.Wq(query).view((Nq, S, self.num_heads, D // self.num_heads))
        k = self.Wk(key).view((Nv, T, self.num_heads, D // self.num_heads))
        v = self.Wv(value).view((Nv, T, self.num_heads, D // self.num_heads))

        attn = (torch.matmul(q.transpose(1, 2), k.permute(0, 2, 3, 1))
                / math.sqrt(D // self.num_heads))  # N, H, S, T
        if attn_mask != None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        score = F.softmax(attn, dim=-1)
        score = self.dropout(score)
        Y = torch.matmul(score, v.transpose(1, 2))
        output = self.A(Y.transpose(1, 2).reshape(Nq, S, D))

        return output
