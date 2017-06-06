import os
from argparse import ArgumentParser

log_template = '\t'.join(
    'Epoch:{:>3.0f},Time:{:>6.0f},Iteration:{:>5.0f},Progress:{:>5.0f}/{:<5.0f},'
    'Loss:{:>8.6f},Acc:{:0.6f}'.split(','))
dev_log_template = '\t'.join('--------- DevMatched/Loss:{:>8.6f},DevMatched/Acc:{:12.4f},DevMisMatched/Loss:{:>8.6f},'
                             'DevMisMatched/Acc:{:12.4f} ---------'.split(','))


def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dev-every', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--save-path', type=str, default='results')
    parser.add_argument('--data-cache', type=str, default=os.path.join(os.getcwd(), '.data_cache'))
    parser.add_argument('--vector-cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word-vectors', type=str, default='glove.840B')
    parser.add_argument('--resume-snapshot', type=str, default='')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--shared-encoder', action='store_true')

    # BiLSTM hyper-params
    parser.add_argument('--d-embed', type=int, default=300)
    parser.add_argument('--d-proj', type=int, default=300)
    parser.add_argument('--d-hidden', type=int, default=300)
    parser.add_argument('--n-layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dp-ratio', type=int, default=0.2)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--train-embed', action='store_false', dest='fix_emb')
    parser.add_argument('--lstm-pooling', type=str, default='none')

    args = parser.parse_args()
    return args
