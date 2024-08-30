import torch
import os
import argparse
import csv
from tools.utils import set_seed, set_save_path, BalancedBatchSizeIterator, save, load_adj, accuracy, EarlyStopping
from tools.run_tools import train_one_epoch_classifier, evaluate_one_epoch_classifier, create_net
from tools.datasets import load_single_subject
from models.NexusNet import NexusNet

def sort_key(s):
    parts = s.split('.')
    part = parts[0].split('_')
    return int(part[-1])

def single_train(args, model, train_X, train_y, val_X, val_y, iterator, index):
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs/4)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=500, max_epochs=args.epochs)
    val_acc = 0

    for epoch in range(0, args.epochs):
        if early_stopper.early_stop:
            print(f'Early stop in epoch {epoch}')
            break

        train_one_epoch_classifier(iterator, (train_X, train_y), model, args.device, optim, criterion)
        scheduler.step()
        avg_acc = evaluate_one_epoch_classifier(iterator, (val_X, val_y), 
                                                model, args.device, criterion)
        
        early_stopper(avg_acc)
        save_checkpoints = {'model_classifier': model.state_dict(),
                            'epoch': epoch + 1,
                            'acc': avg_acc}
        if avg_acc > val_acc:
            val_acc = avg_acc
            save(save_checkpoints, os.path.join(args.model_path, f'{args.id}_{args.exp_type}_{index}.pth.tar'))
    print(f'{args.id}_{args.exp_type}_{index}, val_acc:{val_acc}')
    return model

def single_test(args, model, index, test_X, test_y):
    files = os.listdir(args.model_path)
    pth_files = [file for file in files if file.endswith('.pth.tar')]
    pth_files = sorted(pth_files, key=sort_key)
    pth_file = pth_files[index*9 + args.id-1]

    checkpoint = torch.load(os.path.join(args.model_path, pth_file), map_location=torch.device(args.device))
    model.load_state_dict(checkpoint['model_classifier'])
    model.eval()
    model = model.to(args.device)
    output, _ = model(test_X.to(args.device))
    test_acc, _ = accuracy(output.detach(), test_y.to(args.device).detach())
    test_acc = test_acc[0][0].item()
    print(f'{pth_file}, test_acc:{test_acc}')
    
    with open(f'{args.father_path}/{args.exp_type}_{args.duration}_{args.dataset}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.id, test_acc])

def run(args):
    # ----------------------------------------------environment setting-----------------------------------------------
    set_seed(args.seed)
    args = set_save_path(args.father_path, args)

    # ------------------------------------------------data setting----------------------------------------------------
    train_X, train_y, test_X, test_y, eu_adj = load_single_subject(dataset=args.dataset, subject_id=args.id,
                                                                    to_tensor=True, duration=args.duration)
    val_X = test_X[:len(test_X)//2]
    val_y = test_y[:len(test_y)//2]
    test_X = test_X[len(test_X)//2:]
    test_y = test_y[len(test_y)//2:]
    iterator = BalancedBatchSizeIterator(batch_size=args.batch_size, seed=args.seed)    
    Adj, centrality = load_adj(args.dataset)
    Adj = torch.tensor(Adj, dtype=torch.float32)
    centrality = torch.tensor(centrality, dtype=torch.int64)

    # -----------------------------------------------training setting------------------------------------------------- 
    n_classes =  len(torch.unique(train_y))
    models = create_net(NexusNet, args.exp_type, Adj=Adj, eu_adj=eu_adj, centrality=centrality, in_chans=train_X.shape[1],
                        n_classes=n_classes, input_time_length=train_X.shape[2],
                        drop_prob=args.dropout, pool_mode=args.pool, f1=8, f2=16, kernel_length=64,
                        dataset=args.dataset)
               
    # -------------------------------------------abla_train & train & val--------------------------------------------
    if "train" in args.exp_type:
        print(f"type:{args.exp_type}, target_id:{args.id}, w_decay:{args.w_decay}, dropout:{args.dropout}")
        for i, model in enumerate(models):
            models[i] = single_train(args, model, train_X, train_y, val_X, val_y, iterator, i)

    # -------------------------------------------------test && abla_test-----------------------------------------------------
    for i, model in enumerate(models):
        single_test(args, model, i, test_X, test_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pool', type=str, default='mean', choices=['max', 'mean'])
    parser.add_argument('-dropout', type=float, default=0.25, help='Dropout rate.')
    parser.add_argument('-exp_type', type=str, default="test", 
                        help='train: Train/Val/Test with overall NexusNet.\
                            test: Test using checkpoint.\
                            abla_train: Conduct ablation study with various degraded NexusNet.\
                            abla_test: Test using ablation checkpoint.\
                            duration_train: Train NexusNet with differnet length of MI trials.\
                            duration_test: Test with different length of MI trials.')

    parser.add_argument('-epochs', type=int, default=2000, help='Number of epochs to train.')    
    parser.add_argument('-batch_size', default=64, type=int, help='Batch size.')
    parser.add_argument('-lr', type=float, default=5e-3, help='Learning rate.')
    parser.add_argument('-w_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')

    parser.add_argument('-dataset', type=str, default='bciciv2b', 
                        help='MI Dataset. bciciv2a, bciciv2b')
    parser.add_argument('-duration', type=float, default=4, help='The duration of MI trials.')
    parser.add_argument('-seed', type=int, default='42', help='Random seed.')
    args = parser.parse_args(args=[])

    args.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    if args.exp_type == 'train' or args.exp_type == 'test':
        args.father_path = f'./{args.dataset}_checkpoint'
    elif args.exp_type == 'abla_train' or args.exp_type == 'abla_test':
        args.father_path = f'./{args.dataset}_abla_checkpoint'
    elif args.exp_type == 'duration_train' or args.exp_type == 'duration_test':
        args.father_path = f'./{args.dataset}_duration_checkpoint_{args.duration}'

    if args.dataset=='bciciv2a':
        args.dataset = 'BNCI2014001'
        for id in range(1, 10):
            args.id = id
            run(args)
    elif args.dataset=='bciciv2b':
        args.dataset = 'BNCI2014004'
        for id in range(1, 10):
            args.id = id
            run(args)