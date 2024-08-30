import torch
import math
import sys
from tools.utils import AverageMeter, accuracy

def create_net(model, exp_type, **kwargs):
    models = []
    flag = [1,1,1,1]
    if (exp_type in ['train', 'test', 'duration_train', 'duration_test']):        
        models.append(model(flag, **kwargs))
    elif exp_type == 'single_task':   
        for i in range(4):
            models.append(model(flag, **kwargs))
    else:
        for i in range(0b0000, 0b1111):
            binary_str = bin(i)[2:].zfill(4)
            flag = [int(bit) for bit in binary_str]
            models.append(model(flag, **kwargs))
    return models


def train_one_epoch_classifier(iterator, data, model, device, opt, criterion):
    model.to(device)
    criterion = criterion.to(device)

    model.train()
    data, data_labels = data
    step = 0
    for features, labels in iterator.get_batches(data, data_labels, shuffle=True):
        features = features.to(device)
        labels = labels.to(device)
        logits, _ = model(features)
        loss = criterion(logits, labels)
        if not math.isfinite(loss.item()):
            print(f'In training: loss = {loss.item()}')
            sys.exit(1)
        acc, _ = accuracy(logits.detach(), labels.detach())
        acc = acc[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
        step = step + 1


def evaluate_one_epoch_classifier(iterator, data, model, device, criterion):
    model.to(device)
    model.eval()
    avg_acc = AverageMeter()
    data, data_labels = data
    for features, labels in iterator.get_batches(data, data_labels, shuffle=True):
        features = features.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits, _ = model(features)
            loss = criterion(logits, labels)
        if not math.isfinite(loss.item()):
            print(f'In val: loss = {loss.item()}')
            sys.exit(1)
        acc, _ = accuracy(logits.detach(), labels.detach())
        acc = acc[0]
        avg_acc.update(acc.item(), len(features))
    return avg_acc.avg
