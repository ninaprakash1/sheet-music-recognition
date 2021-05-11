import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from dataset import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from models import *
from tqdm import tqdm





def evaluate(args):
    label_type = args["label_type"]
    emb_dim = args["emb_dim"]
    decoder_dim = args["decoder_dim"]
    att_dim = args["att_dim"]
    dropout = args["dropout"]
    checkpoint = args["checkpoint"]
    data_dir = args["data_dir"]
    label_file = args["label_file"]
    beam_size = args["beam_size"]

    print("model: " + checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (label_type == 'char'):
        corpus, word2idx, max_len = read_captions(label_file)
    elif (label_type == 'word'):
        corpus, word2idx, max_len = read_captions_word(label_file)
    corpus_idx = convert_corpus_idx(word2idx, corpus, max_len)

    decoder = DecoderWithAttention(attention_dim=att_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(word2idx),
                                   dropout=dropout)

    encoder = Encoder()

    assert checkpoint
    checkpoint = torch.load(checkpoint)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    encoder.eval()
    decoder.eval()

    test_loader = torch.utils.data.DataLoader(
        Dataset(data_dir, list(range(0, 4)), corpus_idx),
        batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    # sequences = []
    # ground_truth = []
    counter = 0

    with torch.no_grad():
        for i, (image, caps, caplens) in enumerate(
                tqdm(test_loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

            k = beam_size

            # Move to GPU device, if available
            image = image.to(device)  # (1, 3, 256, 256)

            encoder_out = encoder(image.float())  # (1, enc_image_size, enc_image_size, encoder_dim)
            encoder_dim = encoder_out.size(3)

            encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
            num_pixels = encoder_out.size(1)

            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[word2idx['<start>']]] * k).to(device)  # (k, 1)

            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)

            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()

            # Start decoding
            step = 1
            h, c = decoder.init_hidden_state(encoder_out)
            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:

                embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

                awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe

                h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

                scores = decoder.fc(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // len(word2idx)  # (s)
                next_word_inds = top_k_words % len(word2idx)  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word2idx['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # Break if things have been going on too long
                if step > 50:
                    break
                step += 1

            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
            if seq == caps.squeeze()[:caplens].tolist():
                counter += 1

    # compute exact match
    # EM = accuracy()
    return counter / len(test_loader.dataset)


if __name__ == '__main__':
    args = dict(label_type="word", emb_dim=20, decoder_dim=300, att_dim=300, dropout=0.5,
                batch_size=4,
                workers=0, encoder_lr=1e-4, decoder_lr=1e-4, decay_rate=1, grad_clip=5.0, att_reg=1.0,
                print_freq=100, save_freq=10,
                checkpoint="model/base-2/epoch_2.pt", data_dir="data", label_file="music_strings_small.txt", model_name="base", beam_size=10)
    EM = evaluate(args)
    print("EM: " + str(EM))

