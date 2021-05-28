import os
import numpy as np
import shutil
import queue
import torch
import json
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from torch import tensor


def exact_match(scores, targets):
    scores, targets = scores.detach().numpy(), targets.detach().numpy()
    num_samples, num_classes = scores.shape
    pred = np.array([np.argmax(scores[i]) for i in range(num_samples)])
    num_matches = np.sum(pred == targets)
    return num_matches / num_samples


def scores2string(scores, idx2word):
    scores = scores.detach().numpy()
    preds = np.array([np.argmax(scores[i]) for i in range(scores.shape[0])])
    words = []
    for p in preds:
        words.append(idx2word[int(p)])
    return words

def exact_match_loss(scores, targets):
    scores, targets = scores.detach().numpy(), targets.detach().numpy()
    num_samples, num_classes = scores.shape
    pred = np.array([np.argmax(scores[i]) for i in range(num_samples)])
    num_matches = np.sum(pred == targets)
    return (num_samples - num_matches) / num_samples

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def create_checkpoint_dir(model_name):
    if not os.path.exists("model"):
        os.makedirs("model")
    path = os.path.join("model", model_name)
    if os.path.exists(path):
        i = 1
        while os.path.exists(path + "-" + str(i)):
            i += 1
        path = path + "-" + str(i)
    os.makedirs(path)

    return path

class CheckpointSaver:
    """
    uses a priority queue save the best models

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
    """
    def __init__(self, save_dir, max_checkpoints):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.best_val = None
        self.saved_models = queue.PriorityQueue()

    def save(self, checkpoint_dict, checkpoint_path, metric_val):
        """Save model parameters to disk.
        """
        torch.save(checkpoint_dict, checkpoint_path)
        if not self.best_val:
            self.best_val = metric_val

        if self.best_val <= metric_val:
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)

        priority_order = metric_val

        self.saved_models.put((priority_order, checkpoint_path))
        if self.saved_models.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.saved_models.get()
            try:
                os.remove(worst_ckpt)
                print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                pass

def log_train(writer, train_loss, train_top5, train_top, encoder_lr, decoder_lr, step):
    writer.add_scalar("train/loss", train_loss, step)
    writer.add_scalar("train/top_five_accuracy", train_top5, step)
    writer.add_scalar("train/EM", train_top, step)
    writer.add_scalar("train/encoder_lr", encoder_lr, step)
    writer.add_scalar("train/decoder_lr", decoder_lr, step)
    writer.flush()

def log_val(writer, val_loss, val_top5, val_top, em, pitch, beat, step):
    writer.add_scalar("val/loss", val_loss, step)
    writer.add_scalar("val/top_five_accuracy", val_top5, step)
    writer.add_scalar("val/EM", val_top, step)
    writer.add_scalar("val/true_EM", em, step)
    writer.add_scalar("val/pitch", pitch, step)
    writer.add_scalar("val/beat", beat, step)
    writer.flush()