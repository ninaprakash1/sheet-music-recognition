import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from models import Encoder, DecoderWithAttention
from utils import *
from dataset import *
from torch.utils.tensorboard import SummaryWriter
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

    # make checkpoint path
    path = create_checkpoint_dir(model_name)
    print("checkpoin path: " + str(path))
    writer = SummaryWriter(log_dir=path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (label_type == 'char'):
        corpus, word2idx, max_len = read_captions(label_file)
    elif (label_type == 'word'):
        corpus, word2idx, max_len = read_captions_word(label_file)
    corpus_idx = convert_corpus_idx(word2idx, corpus, max_len)

    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=att_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word2idx),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']

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

    train_loader = torch.utils.data.DataLoader(
        Dataset(data_dir, list(range(0, max_len)), corpus_idx),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(
    # CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
    # batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    for epoch in range(start_epoch, epochs):
        # each training epoch
        decoder.train()  # train mode (dropout and batchnorm is used)
        encoder.train()

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy

        start = time.time()

        # Batches
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

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print(
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                        batch_time=batch_time,
                        loss=losses, top5=top5accs))


        # log to Tensorboard
        writer.add_scalar("train/loss", losses.val, epoch)
        writer.add_scalar("train/top five accuracy", top5accs.val, epoch)
        writer.add_scalar("train/encoder_lr", encoder_optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("train/decoder_lr", decoder_optimizer.param_groups[0]["lr"], epoch)

        writer.flush()


        # TODO evaluate here.

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



def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))

            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    # remember to specify a new model_dir each time. otherwise previous checkpoints will be overwritten
    args = dict(label_type="word", emb_dim=20, decoder_dim=300, att_dim=300, dropout=0.5, start_epoch=0, epochs=120,
                batch_size=32,
                workers=0, encoder_lr=1e-4, decoder_lr=4e-4, decay_rate=0.96, grad_clip=5.0, att_reg=1.0,
                print_freq=100, save_freq=10,
                checkpoint=None, data_dir="data", label_file="music_strings_small.txt", model_name="base")
    main(args)
