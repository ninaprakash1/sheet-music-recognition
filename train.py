from tensorboardX import SummaryWriter
import time
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from models import Encoder, DecoderWithAttention
from utils import *
from dataset import *
import argparse
import sys

PYTHON = sys.executable
# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--label_type', default='char',  # 'char' or 'word'
                    required=False, help='Specify character or word-level prediction')


# TODO an argparse file specifying all default parameters to main().

def main(args):
    """
    Training and validation.
    """

    label_type = args["label_type"]
    emb_dim = args["emb_dim"]
    decoder_dim = args["decoder_dim"]
    att_dim = args["att_dim"]
    dropout = args["dropout"]
    start_epoch = args["start_epoch"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    workers = args["workers"]
    encoder_lr = args["encoder_lr"]
    decoder_lr = args["decoder_lr"]
    decay_rate = args["decay_rate"]
    grad_clip = args["grad_clip"]
    att_reg = args["att_reg"]
    print_freq = args["print_freq"]
    save_freq = args["save_freq"]
    checkpoint = args["checkpoint"]
    data_dir = args["data_dir"]
    label_file = args["label_file"]
    model_name = args["model_name"]
    layers = args["layers"]
    beam_size = args["beam_size"]

    # make checkpoint path
    path = create_checkpoint_dir(model_name)
    print("checkpoin path: " + str(path))
    writer = SummaryWriter(log_dir=path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus, word2idx, max_len = None, None, None
    # pad needs to be the last element of word2idx
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
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),
                                         lr=decoder_lr)
    encoder = Encoder(model_size=int(layers))
    encoder_optimizer = torch.optim.Adam(params=encoder.parameters(),
                                         lr=encoder_lr)

    step = 0
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        step = checkpoint["step"]

    encoder_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=encoder_optimizer, gamma=decay_rate)
    decoder_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=decoder_optimizer, gamma=decay_rate)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # TODO might want to split the data into train, val, test, or we can just generate more test data
    data = Dataset(data_dir, list(range(0, len(corpus))), corpus_idx)

    split = [int(len(data) * 0.8), int(len(data) * 0.2)]
    train_data, val_data = torch.utils.data.dataset.random_split(data, split)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # loader used for beam search for validation
    beam_loader = torch.utils.data.DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    for epoch in range(start_epoch, epochs):

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy
        EM = AverageMeter()

        start = time.time()

        encoder.train()
        decoder.train()

        # Batches
        with torch.enable_grad(), tqdm(total=len(train_data), position=0, leave=True) as progress_bar:
            for i, (imgs, caps, caplens) in enumerate(train_loader):

                data_time.update(time.time() - start)

                # Move to GPU, if available
                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)

                # Forward prop.
                imgs = encoder(imgs.float())
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                # Calculate loss
                loss = criterion(scores.data, targets.data)

                # Add doubly stochastic attention regularization
                loss += att_reg * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # Back prop.
                decoder_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss.backward()

                clip_gradient(decoder_optimizer, grad_clip)
                clip_gradient(encoder_optimizer, grad_clip)

                decoder_optimizer.step()
                encoder_optimizer.step()

                # Keep track of metrics
                top5 = accuracy(scores.data, targets.data, 5)
                losses.update(loss.item(), sum(decode_lengths))
                top5accs.update(top5, sum(decode_lengths))
                batch_time.update(time.time() - start)
                top1 = accuracy(scores.data, targets.data, 1)
                EM.update(top1, sum(decode_lengths))

                start = time.time()

                progress_bar.update(batch_size)
                progress_bar.set_postfix(mode="train", epoch=epoch,
                                         loss=losses.val, Top5=top5accs.val, EM=EM.val)

                step += batch_size

                # Print status
                if i != 0 and i % print_freq == 0:
                    print('Train\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'
                          'EM {EM.val:.3f}'.format(
                        batch_time=batch_time,
                        loss=losses, top5=top5accs, EM=EM))

                # log to Tensorboard
                writer.add_scalar("train/loss", losses.val, step)
                writer.add_scalar("train/top five accuracy", top5accs.val, step)
                writer.add_scalar("train/EM", EM.val, step)
                writer.add_scalar("train/encoder_lr", encoder_optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("train/decoder_lr", decoder_optimizer.param_groups[0]["lr"], step)
                writer.flush()

        val_loss, val_top5, val_top, em = validate(val_loader, beam_loader, encoder, decoder, criterion,
                                               device, att_reg, epoch, beam_size=beam_size, word2idx=word2idx)
        writer.add_scalar("val/loss", val_loss, step)
        writer.add_scalar("val/top five accuracy", val_top5, step)
        writer.add_scalar("val/EM", val_top, step)
        writer.add_scalar("val/true EM", em, step)
        print('\nValidation\t'
              'Loss {loss:.4f}\t'
              'top5 {top5:.3f}\t'
              'EM {top:.3f}\ttrue EM {em:.3f}'.format(
            loss=val_loss, top5=val_top5, top=val_top, em=em))


        if epoch > 0 and epoch % save_freq == 0:
            checkpoint_path = os.path.join(path, f'epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'loss': losses.val,
            }, checkpoint_path)

            print(f'Saved checkpoint: {checkpoint_path}')

        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()


def validate(val_loader, beam_loader, encoder, decoder, criterion, device, att_reg, epoch, beam_size=10, word2idx=None):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()  # top5 accuracy
    topacc = AverageMeter()

    start = time.time()

    encoder.eval()
    decoder.eval()

    with torch.no_grad(), \
            tqdm(total=len(val_loader.dataset), position=0, leave=True) as progress_bar:
        for i, (imgs, caps, caplens) in enumerate(val_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            imgs = encoder(imgs.float())
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]

            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores.data, targets.data)

            # Add doubly stochastic attention regularization
            loss += att_reg * ((1. - alphas.sum(dim=1)) ** 2).mean()
            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            progress_bar.update(val_loader.batch_size)
            progress_bar.set_postfix(mode="val", epoch=epoch, NLL=losses.val)
            top5 = accuracy(scores.data, targets.data, 5)
            top5accs.update(top5, sum(decode_lengths))
            top = accuracy(scores.data, targets.data, 1)
            topacc.update(top, sum(decode_lengths))

    # start computing true EM
    counter = 0
    with torch.no_grad():
        for i, (image, caps, caplens) in enumerate(
                tqdm(beam_loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size), position=0, leave=True)):

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

            if len(complete_seqs) > 0:
                i = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[i]
            else:
                i = int(scores.argmax() // scores.shape[1])
                seq = seqs[i].tolist()
            if seq == caps.squeeze()[:caplens].tolist():
                counter += 1

    return losses.val, top5accs.val, topacc.val, (counter / len(beam_loader.dataset))


if __name__ == '__main__':
    args = dict(label_type="word", emb_dim=20, decoder_dim=300, att_dim=300, dropout=0.5, start_epoch=0, epochs=120,
                batch_size=32,
                workers=0, encoder_lr=1e-4, decoder_lr=4e-4, decay_rate=0.96, grad_clip=5.0, att_reg=1.0,
                print_freq=20, save_freq=10,
                checkpoint=None, data_dir="data", label_file="music_strings_small.txt", model_name="base", layers=34,
                beam_size=10)
    main(args)
