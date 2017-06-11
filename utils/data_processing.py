import numpy as np
import re
import pickle
import os
import random
import json
import collections
from multiprocessing import Pool

import time
from ufal.udpipe import Model, Pipeline, ProcessingError

import parameters as params

FIXED_PARAMETERS = params.load_parameters()

LABEL_MAP = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2,
    'hidden': 0
}

PADDING = '<PAD>'
UNKNOWN = '<UNK>'

ID, FORM, LEMMA, UPOSTAG, XPOSTAG, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)
FIRST_CLASS_POS = {'ADJ', 'ADV', 'NOUN', 'VERB'}


def process_dep(data, seq_length, r):
    parsed_table = [row.split('\t') for row in data.split('\n') if len(row) > 0 and row[0] != u'#']
    waiting = collections.deque([word[ID] for word in parsed_table if word[HEAD] == u'0'])
    ret = np.zeros((r, seq_length), 'float32')
    i = 0
    while i < r:
        if len(waiting) <= 0:
            break
        cur = waiting.popleft()
        cur_cluster = [word[ID] for word in parsed_table if word[HEAD] == cur]
        waiting.extend(cur_cluster)

        cur_cluster.append(cur)
        vl = len(cur_cluster)
        if vl < 2:
            continue
        vl = 1. / vl
        for word_id in cur_cluster:
            real_id = int(word_id) - 1
            if real_id < seq_length:
                ret[i][real_id] = vl
        i += 1
    return ret


def process_line(line):
    loaded_example = json.loads(line)
    if loaded_example['gold_label'] not in LABEL_MAP:
        return None
    loaded_example['label'] = LABEL_MAP[loaded_example['gold_label']]
    if is_snli:
        loaded_example['genre'] = 'snli'

    if pipeline:
        processed = pipeline.process(loaded_example['sentence1'], error)
        if error.message != '':
            return None
        loaded_example['prem_dep'] = process_dep(processed, pr_seq_length, pr_r)
        processed = pipeline.process(loaded_example['sentence2'], error)
        if error.message != '':
            return None
        loaded_example['hypo_dep'] = process_dep(processed, pr_seq_length, pr_r)

    return loaded_example


def load_nli_data(path, snli=False, udpipe_path=None, seq_length=50, r=10, cache_file=''):
    """
    Load MultiNLI or SNLI data.
    If the 'snli' parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    global is_snli, pipeline, error, pr_seq_length, pr_r
    is_snli = snli
    pr_r = r
    pr_seq_length = seq_length
    print(path)
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = [w for w in pickle.load(f) if w is not None]
    else:
        if udpipe_path:
            model = Model.load(udpipe_path)
            pipeline = Pipeline(model, 'horizontal', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
            error = ProcessingError()

        with open(path) as f:
            pool = Pool(32)
            data = pool.map_async(process_line, list(f), chunksize=1)
            while not data.ready():
                print('{} lines left'.format(data._number_left))
                time.sleep(10)
            data = [w for w in data.get() if w is not None]
            pool.close()
            random.seed(1)
            random.shuffle(data)

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    return data


def load_nli_data_genre(path, genre, snli=True):
    """
    Load a specific genre's examples from MultiNLI, or load SNLI data and assign a 'snli' genre to the examples.
    If the 'snli' parameter is set to True, a genre label of snli will be assigned to the data. If set to true, it will
    overwrite the genre label for MultiNLI data.
    """
    data = []
    j = 0
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example['gold_label'] not in LABEL_MAP:
                continue
            loaded_example['label'] = LABEL_MAP[loaded_example['gold_label']]
            if snli:
                loaded_example['genre'] = 'snli'
            if loaded_example['genre'] == genre:
                data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data


def tokenize(string):
    string = re.sub(r'\(|\)', '', string)
    return string.split()


def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['sentence1_binary_parse']))
            word_counter.update(tokenize(example['sentence2_binary_parse']))

    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary

    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices


def sentences_to_padded_index_sequences(word_indices, datasets):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    for _id, dataset in enumerate(datasets):
        for example in dataset:
            for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
                example[sentence + '_index_sequence'] = np.zeros((FIXED_PARAMETERS['seq_length']), dtype=np.int32)

                token_sequence = tokenize(example[sentence])
                padding = FIXED_PARAMETERS['seq_length'] - len(token_sequence)

                for i in range(FIXED_PARAMETERS['seq_length']):
                    if i >= len(token_sequence):
                        index = word_indices[PADDING]
                    else:
                        if token_sequence[i] in word_indices:
                            index = word_indices[token_sequence[i]]
                        else:
                            index = word_indices[UNKNOWN]
                    example[sentence + '_index_sequence'][i] = index


def load_embedding_zeros(path, word_indices):
    """
    Load GloVe embeddings. Initializng OOV words to vector of zeros.
    """
    emb = np.zeros((len(word_indices), FIXED_PARAMETERS['word_embedding_dim']), dtype='float32')

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if FIXED_PARAMETERS['embeddings_to_load'] is not None:
                if i >= FIXED_PARAMETERS['embeddings_to_load']:
                    break

            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb


def load_embedding_rand(path, word_indices):
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    n = len(word_indices)
    m = FIXED_PARAMETERS['word_embedding_dim']
    emb = np.empty((n, m), dtype=np.float32)

    emb[:, :] = np.random.normal(size=(n, m))

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0:2, :] = np.zeros((1, m), dtype='float32')

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if FIXED_PARAMETERS['embeddings_to_load'] is not None:
                if i >= FIXED_PARAMETERS['embeddings_to_load']:
                    break

            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb
