
import time
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from models import ResNet, RNN_Decoder, SqueezeNet
from utils import *
from dataset import *
from eval import idx2string, pitch_match, beat_match
from transformer_decoder import CaptioningTransformer, ImageEncoder
from tensorboardX import SummaryWriter


# import argparse
# import sys

# PYTHON = sys.executable
# # Set up command line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('-l', '--label_type', default='char',  # 'char' or 'word'
#                     required=False, help='Specify character or word-level prediction')


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
    train_dir = args["train_dir"]
    val_dir = args["val_dir"]
    train_label = args["train_label"]
    val_label = args["val_label"]
    model_name = args["model_name"]
    beam_size = args["beam_size"]
    backbone = args["backbone"]
    decode_type = args["decode_type"]
    spatial_encode = args["spatial_encode"]
    transformer_encode = args["transformer_encode"]

    # make checkpoint path
    path = create_checkpoint_dir(model_name)
    print("checkpoin path: " + str(path))
    writer = SummaryWriter(log_dir=path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus, word2idx, max_len = None, None, None
    # pad needs to be the last element of word2idx
    if (label_type == 'char'):
        train_corpus, word2idx, max_len = read_captions(train_label)
        val_corpus, _, val_max_len = read_captions(val_label)
    elif (label_type == 'word'):
        train_corpus, word2idx, idx2word, max_len = read_captions_word(train_label)
        val_corpus, _, _, val_max_len = read_captions_word(val_label)
    if val_max_len > max_len:
        max_len = val_max_len
    train_corpus_idx = convert_corpus_idx(word2idx, train_corpus, max_len)
    val_corpus_idx = convert_corpus_idx(word2idx, val_corpus, max_len)

    encoder, decoder = None, None
    if decode_type == "RNN":
        decoder = RNN_Decoder(attention_dim=att_dim, embed_dim=emb_dim, decoder_dim=decoder_dim,
                              word2idx=word2idx, dropout=dropout)
        encoder = ImageEncoder(backbone=backbone, wordvec_dim=emb_dim, transformer_encode=False, RNN_decode=True,
                               spatial_encode=False)
    elif decode_type == "Transformer":
        decoder = CaptioningTransformer(word2idx=word2idx, wordvec_dim=emb_dim, input_dim=emb_dim, max_length=max_len+2, num_layers=4)
        encoder = ImageEncoder(backbone=backbone, wordvec_dim=emb_dim, num_layers=4, transformer_encode=transformer_encode,
                               spatial_encode=spatial_encode, RNN_decode=False)

    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=decoder_lr)
    encoder_optimizer = torch.optim.Adam(params=encoder.parameters(),
                                         lr=encoder_lr)

    encoder_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=encoder_optimizer, gamma=decay_rate)
    decoder_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=decoder_optimizer, gamma=decay_rate)

    checkpoint_saver = CheckpointSaver(path, 5)

    step = 0
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch']
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        encoder_lr_scheduler.load_state_dict(checkpoint["encoder_lr_scheduler_state_dict"])
        decoder_lr_scheduler.load_state_dict(checkpoint["decoder_lr_scheduler_state_dict"])
        step = checkpoint["step"]

    optimizer_to(decoder_optimizer, device)
    optimizer_to(encoder_optimizer, device)


    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    # val_idx = list(range(1, 20000, 20))
    # test_idx = list(range(0, 20000, 20))
    # train_idx = list(set(list(range(0, 22000))) - set(val_idx) - set(test_idx))

    train_idx = list(set(range(7500)) - set(range(0, 7500, 15)))
    val_idx = list(range(0, 7500, 15))
    train_data = Dataset(train_dir, train_idx, train_corpus_idx)
    val_data = Dataset(val_dir, val_idx, val_corpus_idx)

    # split = [500, 3, len(train_data)-503]
    # split = [len(data)-500, 500, 0]
    # split = [1, 6999]
    # train_data, _, rest = torch.utils.data.dataset.random_split(train_data, split)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # loader used for beam search for validation
    beam_loader = torch.utils.data.DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)

    for epoch in range(start_epoch+1, epochs+1):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top5accs = AverageMeter()
        EM = AverageMeter()

        start = time.time()

        encoder.train()
        decoder.train()

        # Batches
        with torch.enable_grad(), tqdm(total=len(train_data), position=0, leave=True) as progress_bar:
            for i, (imgs, caps, caplens) in enumerate(train_loader):

                data_time.update(time.time() - start)

                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)

                imgs = encoder(imgs.float())

                if decode_type == "RNN":
                    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, device)
                    targets = caps_sorted[:, 1:]
                elif decode_type == "Transformer":
                    scores = decoder(imgs, caps[:, 0:-1])
                    targets = caps[:, 1:]

                # no need to decode at <end>
                decode_lengths = (caplens.squeeze(1) - 1).tolist()

                # more efficient computation
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=False)
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)

                loss = criterion(scores.data, targets.data)

                if decode_type == "RNN":
                    loss += att_reg * ((1. - alphas.sum(dim=1)) ** 2).mean()

                decoder_optimizer.zero_grad()
                encoder_optimizer.zero_grad()
                loss.backward()

                clip_gradient(decoder_optimizer, grad_clip)
                clip_gradient(encoder_optimizer, grad_clip)

                decoder_optimizer.step()
                encoder_optimizer.step()

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

                if i != 0 and i % print_freq == 0:
                    print('Train\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'
                          'EM {EM.val:.3f}'.format(
                        batch_time=batch_time,
                        loss=losses, top5=top5accs, EM=EM))

                # log to Tensorboard
                log_train(writer, losses.val, top5accs.val, EM.val, encoder_optimizer.param_groups[0]["lr"],
                          decoder_optimizer.param_groups[0]["lr"], step)
        if epoch % 2 == 0:
            val_loss, val_top5, val_top, em, pitch, beat = validate(val_loader, beam_loader, encoder, decoder, criterion,
                                                   device, att_reg, epoch, decode_type, beam_size=beam_size, idx2word=idx2word)

            log_val(writer, val_loss, val_top5, val_top, em, pitch, beat, step)

            print('\nValidation\t'
                  'Loss {loss:.4f}\t'
                  'top5 {top5:.3f}\t'
                  'EM {top:.3f}\t'
                  'true EM {em:.3f}\t'
                  'pitch {pitch: .3f}\t'
                  'beat {beat: .3f}'.format(
                loss=val_loss, top5=val_top5, top=val_top, em=em, pitch=pitch, beat=beat))

            if epoch % save_freq == 0:
                checkpoint_path = os.path.join(path, f'epoch_{epoch}.pt')
                checkpoint_dict = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                    'decoder_lr_scheduler_state_dict': decoder_lr_scheduler.state_dict(),
                    'encoder_lr_scheduler_state_dict': encoder_lr_scheduler.state_dict(),
                    'loss': losses.val,
                    'step': step
                }
                checkpoint_saver.save(checkpoint_dict, checkpoint_path, em)
                print(f'Saved checkpoint: {checkpoint_path}')

        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()


def validate(val_loader, beam_loader, encoder, decoder, criterion, device, att_reg, epoch, decode_type, beam_size=10, idx2word=None):
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
            if decode_type == "RNN":
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, device)
                targets = caps_sorted[:, 1:]
            elif decode_type == "Transformer":
                scores = decoder(imgs, caps[:, 0:-1])
                targets = caps[:, 1:]
            # no need to decode at <end>
            decode_lengths = (caplens.squeeze(1) - 1).tolist()


            # more efficient computation
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=False)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)

            loss = criterion(scores.data, targets.data)

            if decode_type == "RNN":
                loss += att_reg * ((1. - alphas.sum(dim=1)) ** 2).mean()

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
    pitch_match_score = 0
    beat_match_score = 0

    for i, (image, caps, caplens) in enumerate(
                         tqdm(beam_loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size), position=0, leave=True)):

        image = image.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        image = encoder(image.float())
        image = image.squeeze(1)
        seq = decoder.sample(image, device, beam_size=10)
        if seq == caps.squeeze()[:caplens].tolist():
            counter += 1

        pred_seq = idx2string(seq, idx2word)
        target_seq = idx2string(caps.squeeze()[:caplens].tolist(), idx2word)

        pitch_match_score += pitch_match(pred_seq, target_seq)
        beat_match_score += beat_match(pred_seq, target_seq)

    true_EM = counter / len(val_loader.dataset)
    pitch_match_score /= len(val_loader.dataset)
    beat_match_score /= len(val_loader.dataset)

    return losses.val, top5accs.val, topacc.val, true_EM, pitch_match_score, beat_match_score


def optimizer_to(optim, device):
    """
    move optimizer to device.
    ref: https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

if __name__ == '__main__':
    args = dict(label_type="word", emb_dim=200, decoder_dim=300, att_dim=300, dropout=0.2, start_epoch=0, epochs=200,
                batch_size=16,
                workers=2, encoder_lr=0.0001, decoder_lr=0.0001, decay_rate=0.98, grad_clip=5.0, att_reg=1.0,
                print_freq=100, save_freq=1,
                backbone="squeezenet", # [resnet18, resnet34, squeezenet]
                checkpoint=None, train_dir="different_measures", val_dir="different_measures",
                train_label="different_measures_strings.txt", val_label="different_measures_strings.txt", model_name="deep_encoder",
                beam_size=10, decode_type="Transformer", # [RNN, transformer]
                spatial_encode=True, transformer_encode=True
                )
    main(args)
