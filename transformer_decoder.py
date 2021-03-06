"""
Adapted from problem set 3.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import copy
from models import SqueezeNet, ResNet


class ImageEncoder(nn.Module):
    """
    Transformer Encoder
    """
    def __init__(self, backbone="squeezenet", wordvec_dim=200, num_heads=4,
                 num_layers=2, transformer_encode=True, RNN_decode=False, spatial_encode=True):

        super().__init__()
        self.transformer_encode = transformer_encode
        self.RNN_decode = RNN_decode
        self.spatial_encode = spatial_encode
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
        if not RNN_decode:
            self.proj = nn.Linear(512, wordvec_dim)


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
            if module.bias != None:
                module.bias.data.zero_()
            nn.init.xavier_uniform_(module.weight.data)

    def forward(self, features):
        x = self.backbone(features)
        if self.RNN_decode:
            return x
        x = self.proj(x)
        if self.spatial_encode:
            x = self.positional_encoding(x)
        if self.transformer_encode:
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
    def __init__(self, word2idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=2, max_length=50, ensemble=False):
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

        self._null = word2idx["<pad>"]
        self._start = word2idx.get("<start>", None)
        self._end = word2idx.get("<end>", None)
        vocab_size = len(word2idx)
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.num_heads = num_heads
        self.num_layers = num_layers

        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim)

        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)

        self.ensemble = ensemble
        if self.ensemble:
            self.output = nn.ModuleList([nn.Linear(wordvec_dim, vocab_size) for i in range(num_layers)])
        else:
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

        if self.ensemble:
            # taking the average of the scores before softmax. adding superversion to each layer
            scores = 0
            for i in range(self.num_layers):
                scores = scores + self.output[i](features[i])
            scores = scores / self.num_layers
        else:
            # Project to scores per token.
            # shape: [N, T, W] -> [N, T, V], V vocab size
            scores = self.output(features[-1])

        return scores

    def sample(self, features, device, max_decode_length=40, beam_size=10):
        """
        Given image features, use greedy decoding to predict the image caption.

        Inputs:
         - features: image features, of shape (1, D, T')
         - max_length: maximum possible caption length

        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """


        with torch.no_grad():
            k = beam_size

            k_prev_words = torch.LongTensor([[self._start]] * k).to(device) # (k, 1)

            seqs = k_prev_words  # (k, 1)

            top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

            complete_seqs = list()
            complete_seqs_scores = list()

            # start with the token after <start>
            for step in range(1, max_decode_length+1):

                # features = features.repeat(k, 1, 1)
                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, seqs)
                output_logits = output_logits[:, -1, :] # [k, V]

                scores = F.log_softmax(output_logits, dim=1)
                scores = top_k_scores.expand_as(scores) + scores

                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
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

            return seq

    def greedy_decode(self, features, device, max_length=35):
        """
        a differentiable implementation of greedy decoding

        return:
            one_hots: greedy decoding result for each time step
            caps_len: the length of each decoded sequence (including <start> and <end>)
        """

        N = features.shape[0]

        # Create an empty captions tensor (where all tokens are NULL).
        captions = self._null * torch.ones((N, max_length), dtype=torch.long).to(device)

        # Create a partial caption, with only the start token.
        partial_caption = torch.LongTensor([[self._start]] * N).to(device)
        captions[:, 0] = partial_caption.squeeze()
        # [N] -> [N, 1]
        incomplete_inds = list(range(N))
        caps_len = list([0] * N)
        one_hots = self._null * torch.ones((N, max_length-1, 734)).to(device)

        for t in range(1, max_length):
            # Predict the next token (ignoring all other time steps).
            output_logits = self.forward(features, partial_caption)
            output_logits = output_logits[:, -1, :]

            one_hot = F.gumbel_softmax(output_logits, tau=1, hard=True)
            one_hots[incomplete_inds, t-1, :] = one_hot[incomplete_inds, :]
            _, word = torch.where(one_hot == 1)

            captions[incomplete_inds, t] = word[incomplete_inds]
            complete = [ind for ind, next_word in enumerate(word.squeeze()) if
                               next_word == self._end]

            new_complete= list((set(incomplete_inds) - (set(incomplete_inds) - set(complete))))
            incomplete_inds = list(set(incomplete_inds) - set(complete))

            for idx in new_complete:
                caps_len[idx] = t+1

            partial_caption = captions[:, :t+1].clone()
            partial_caption = partial_caption.reshape((N, -1))
            if (len(incomplete_inds) == 0):
                break
        if(len(incomplete_inds) > 0):
            for idx in incomplete_inds:
                caps_len[idx] = max_length

        return one_hots, caps_len

    def reinforce(self, features, device, max_decode_length=34, beam_size=5):
        """
        let gradient pass through softmax only
        """

        k = beam_size

        k_prev_words = torch.LongTensor([[self._start]] * k).to(device) # (k, 1)

        seqs = k_prev_words  # (k, 1)

        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        complete_seqs = list()
        scores_tensor = torch.zeros(beam_size).to(device)

        # start with the token after <start>
        for step in range(1, max_decode_length+1):

            output_logits = self.forward(features, seqs)
            output_logits = output_logits[:, -1, :] # [k, V]

            scores = F.log_softmax(output_logits, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                _, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                _, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            top_k_scores = scores.view(-1)[top_k_words]
            prev_word_inds = top_k_words // self.vocab_size  # (s)
            next_word_inds = top_k_words % self.vocab_size  # (s)

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1).to(device)

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != self._end]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds])
                scores_tensor[beam_size-k:beam_size-k+len(complete_inds)] = top_k_scores[complete_inds]
            k -= len(complete_inds)

            if k == 0:
                break

            seqs = seqs[incomplete_inds]

            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)

        if len(complete_seqs) == beam_size:
            complete_seqs = torch.nn.utils.rnn.pad_sequence(complete_seqs, batch_first=True, padding_value=self._null)
            complete_seqs = torch.cat([complete_seqs.to(device), self._null * torch.ones(beam_size, 35-complete_seqs.shape[1]).to(device)], dim=1).to(device)
            return complete_seqs, scores_tensor
        else:
            complete_seqs.extend(seqs)
            scores_tensor[beam_size-k:] = top_k_scores.squeeze(1)
            complete_seqs = torch.nn.utils.rnn.pad_sequence(complete_seqs, batch_first=True, padding_value=self._null)

        return complete_seqs, scores_tensor

    def compute_policy_gradient_batch(self, sample, tgt, scores, caps_len):
        with torch.no_grad():
            reward, baseline = self.compute_baseline(sample, tgt, caps_len)
        loss = (- (reward - baseline) * scores).mean()
        return loss


    def compute_baseline(self, gen, tgt, caps_len):
        B, T = gen.shape
        caps_len = torch.tensor(caps_len).reshape((B, -1))

        indicator = gen.eq(tgt)
        reward = indicator.sum(axis=1).float()
        reward = reward - T + caps_len
        baseline = reward.mean()


        return reward, baseline


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
        output_seq = []
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)
            output_seq.append(output)

        return output_seq

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


class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """

    def __init__(self, embed_dim, dropout=0.1, max_len=1000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0

        pe = torch.zeros(1, max_len, embed_dim)

        even = torch.arange(0, embed_dim, 2)
        even_mat = torch.tensor(list(range(0, max_len))).unsqueeze(1).repeat(1, len(even))
        coef = torch.exp(math.log(10000) * -1 * even / embed_dim)
        even_mat = even_mat * coef

        pe[0, :, 0::2] = torch.sin(even_mat)
        pe[0, :, 1::2] = torch.cos(even_mat)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape

        output = x + self.pe[:, :S, :]
        output = self.dropout(output)

        return output





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
