import os
import time
import glob

import torch
import torch.optim as optim
import torch.nn as nn
from torchtext import data

from models.bilstm import TheModel
from utils import multinli
from utils.misc import get_args, dev_log_template, log_template

args = get_args()
torch.cuda.set_device(args.gpu)


def train():
    inputs = data.Field(lower=args.lower)
    answers = data.Field(sequential=False)
    train_set, dev_matched_set, test_matched_set = multinli.MultiNLI.splits(inputs, answers)
    inputs.build_vocab(train_set, dev_matched_set, test_matched_set)
    if args.word_vectors:
        if os.path.isfile(args.vector_cache):
            inputs.vocab.vectors = torch.load(args.vector_cache)
        else:
            inputs.vocab.load_vectors(wv_dir=args.data_cache, wv_type=args.word_vectors, wv_dim=args.d_embed)
            os.makedirs(os.path.dirname(args.vector_cache))
            torch.save(inputs.vocab.vectors, args.vector_cache)
    answers.build_vocab(train_set)
    train_iter, dev_matched_iter, test_matched_iter = data.BucketIterator.splits(
        (train_set, dev_matched_set, test_matched_set), batch_size=args.batch_size, device=args.gpu)
    config = args
    config.n_embed = len(inputs.vocab)
    config.d_out = len(answers.vocab)
    config.n_cells = config.n_layers
    if config.birnn:
        config.n_cells *= 2
    if args.resume_snapshot:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        model = TheModel(config)
        if args.word_vectors:
            model.embed.weight.data = inputs.vocab.vectors
            if args.gpu >= 0:
                model.cuda()
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    iterations = 0
    start = time.time()
    best_dev_matched_acc = -1
    train_iter.repeat = False
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for epoch in range(args.epochs):
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(train_iter):
            model.train()
            opt.zero_grad()
            iterations += 1
            answer = model(batch)
            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
            n_total += batch.batch_size
            train_acc = 100. * n_correct / n_total
            loss = criterion(answer, batch.label)
            loss.backward()
            opt.step()
            if iterations % args.save_every == 0:
                snapshot_prefix = os.path.join(args.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc,
                                                                                                    loss.data[0],
                                                                                                    iterations)
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

            if iterations % args.log_every == 0:
                print(log_template.format(epoch, time.time() - start, iterations, 1 + batch_idx, len(train_iter),
                                          loss.data[0], n_correct * 100. / n_total))

            if iterations % args.dev_every == 0:
                dev_matched_acc, dev_matched_loss = model_eval_dev(criterion, dev_matched_iter, dev_matched_set, model)
                print(dev_log_template.format(dev_matched_loss.data[0], dev_matched_acc))
                if dev_matched_acc > best_dev_matched_acc:
                    best_dev_matched_acc = dev_matched_acc
                    snapshot_prefix = os.path.join(args.save_path, 'best_snapshot_matched')
                    snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}_iter_{}_model.pt'.format(
                        dev_matched_acc, dev_matched_loss.data[0], iterations)
                    torch.save(model, snapshot_path)
                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != snapshot_path:
                            os.remove(f)


def model_eval_dev(criterion, dev_matched_iter, dev_matched_set, model):
    model.eval()
    dev_matched_iter.init_epoch()
    n_dev_correct, dev_loss = 0, 0
    for dev_batch_idx, dev_batch in enumerate(dev_matched_iter):
        answer = model(dev_batch)
        n_dev_correct += (
            torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
        dev_loss = criterion(answer, dev_batch.label)
    dev_acc = 100. * n_dev_correct / len(dev_matched_set)
    return dev_acc, dev_loss


train()
