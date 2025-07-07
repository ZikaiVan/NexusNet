import numpy as np
import torch
import random
import errno
import os
import sys
from numpy.random import RandomState

class BalancedBatchSizeIterator(object):
    """
    Create batches of balanced size.
    
    Parameters
    ----------
    batch_size: int
        Resulting batches will not necessarily have the given batch size
        but rather the next largest batch size that allows to split the set into
        balanced batches (maximum size difference 1).
    seed: int
        Random seed for initialization of `numpy.RandomState` random generator
        that shuffles the batches.
    """

    def __init__(self, batch_size, seed=328774):
        self.batch_size = batch_size
        self.seed = seed
        self.rng = RandomState(self.seed)

    def get_batches(self, X, y, shuffle):
        n_trials = len(X)
        batches = get_balanced_batches(
            n_trials, batch_size=self.batch_size, rng=self.rng, shuffle=shuffle
        )
        self.len = len(batches)
        for batch_inds in batches:
            batch_X = X[batch_inds]
            batch_y = y[batch_inds]

            # add empty fourth dimension if necessary
            # if batch_X.ndim == 3:
            #     batch_X = batch_X[:, :, :, None]
            yield (batch_X, batch_y)

    def reset_rng(self):
        self.rng = RandomState(self.seed)

    def __len__(self):
        return self.len

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def set_save_path(father_path, args):
    if not os.path.exists(father_path):
        os.makedirs(father_path)
    args.model_adj_path = father_path
    args.model_path = father_path
    return args

def save(checkpoints, save_path):
    torch.save(checkpoints, save_path)

def get_balanced_batches(
    n_trials, rng, shuffle, n_batches=None, batch_size=None
):
    """Create indices for batches balanced in size 
    (batches will have maximum size difference of 1).
    Supply either batch size or number of batches. Resulting batches
    will not have the given batch size but rather the next largest batch size
    that allows to split the set into balanced batches (maximum size difference 1).

    Parameters
    ----------
    n_trials : int
        Size of set.
    rng : RandomState
    shuffle : bool
        Whether to shuffle indices before splitting set.
    n_batches : int, optional
    batch_size : int, optional

    Returns
    -------

    """
    assert batch_size is not None or n_batches is not None
    if n_batches is None:
        n_batches = int(np.round(n_trials / float(batch_size)))

    if n_batches > 0:
        min_batch_size = n_trials // n_batches
        n_batches_with_extra_trial = n_trials % n_batches
    else:
        n_batches = 1
        min_batch_size = n_trials
        n_batches_with_extra_trial = 0
    assert n_batches_with_extra_trial < n_batches
    all_inds = np.array(range(n_trials))
    if shuffle:
        rng.shuffle(all_inds)
    i_start_trial = 0
    i_stop_trial = 0
    batches = []
    for i_batch in range(n_batches):
        i_stop_trial += min_batch_size
        if i_batch < n_batches_with_extra_trial:
            i_stop_trial += 1
        batch_inds = all_inds[range(i_start_trial, i_stop_trial)]
        batches.append(batch_inds)
        i_start_trial = i_stop_trial
    assert i_start_trial == n_trials
    return batches


def load_adj(dn='bciciv2a', norm=False):
    if 'BNCI2014001' == dn:
        num_node = 22
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(1, 3), (1, 4), (1, 5),
                         (2, 3), (2, 7), (2, 8), (2, 9),
                         (3, 4), (3, 8), (3, 9), (3, 10),
                         (4, 5), (4, 9), (4, 10), (4, 11),
                         (5, 6), (5, 10), (5, 11), (5, 12),
                         (6, 11), (6, 12), (6, 13),
                         (7, 8), (7, 14),
                         (8, 9), (8, 14), (8, 15),
                         (9, 10), (9, 14), (9, 15), (9, 16),
                         (10, 11), (10, 15), (10, 16), (10, 17),
                         (11, 12), (11, 16), (11, 17), (11, 18),
                         (12, 13), (12, 17), (12, 18),
                         (13, 18),
                         (14, 15), (14, 19),
                         (15, 16), (15, 19), (15, 20),
                         (16, 17), (16, 19), (16, 20), (16, 21),
                         (17, 18), (17, 20), (17, 21),
                         (18, 21),
                         (19, 20), (19, 22),
                         (20, 21), (20, 22),
                         (21, 22)]
    elif 'BNCI2014004' == dn:
        num_node = 3
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(1, 2), (2, 3)]

    centrality = np.zeros(num_node, dtype=np.int64)
    for pair in neighbor_link:
        centrality[pair[0] - 1] += 1
        centrality[pair[1] - 1] += 1
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link]
    edge = self_link + neighbor_link
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1
        
    return A, centrality


def accuracy(output, target, topk=(1,)):
    shape = None
    if 2 == len(target.size()):
        shape = target.size()
        target = target.view(target.size(0))
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    if shape:
        target = target.view(shape)
    return ret, pred


class Logger(object):
    def __init__(self, fpath):
        self.console = sys.stdout
        self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


class EarlyStopping(object):
    """
    Early stops the training if validation loss
    doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, max_epochs=80):
        """
        patience (int): How long to wait after last time validation
        loss improved.
                        Default: 7
        verbose (bool): If True, prints a message for each validation
        loss improvement.
                        Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.max_epochs = max_epochs
        self.max_epoch_stop = False
        self.epoch_counter = 0
        self.should_stop = False
        self.checkpoint = None

    def __call__(self, val_loss):
        self.epoch_counter += 1
        if self.epoch_counter >= self.max_epochs:
            self.max_epoch_stop = True

        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        if any([self.max_epoch_stop, self.early_stop]):
            self.should_stop = True
