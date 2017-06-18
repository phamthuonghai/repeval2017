"""
Script to generate a CSV file of predictions on the test data.
"""

import os
import importlib
import pickle

import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from xgboost.sklearn import XGBClassifier
from tqdm import tqdm

import utils.parameters as params
from utils import logger
from utils.data_processing import *
from utils.evaluate import *

FIXED_PARAMETERS = params.load_parameters()
modname = FIXED_PARAMETERS["model_name"]
logpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".log"
logger = logger.Logger(logpath)

model = FIXED_PARAMETERS["model_type"]

module = importlib.import_module(".".join(['models', model]))
MyModel = getattr(module, 'MyModel')

# Logging parameter settings at each launch of training script
# This will help ensure nothing goes awry in reloading a model and we consistenyl use the same hyperparameter settings.
logger.log("FIXED_PARAMETERS\n %s" % FIXED_PARAMETERS)

logger.log("Loading data")
training_mnli = load_nli_data(FIXED_PARAMETERS["training_mnli"], udpipe_path=FIXED_PARAMETERS['udpipe_path'],
                              seq_length=FIXED_PARAMETERS['seq_length'], r=FIXED_PARAMETERS['s2_dim'],
                              cache_file=os.path.join(FIXED_PARAMETERS["log_path"], modname)+'.training_mnli.cache')
dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"], udpipe_path=FIXED_PARAMETERS['udpipe_path'],
                            seq_length=FIXED_PARAMETERS['seq_length'], r=FIXED_PARAMETERS['s2_dim'],
                            cache_file=os.path.join(FIXED_PARAMETERS["log_path"], modname)+'.dev_matched.cache')
test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"], udpipe_path=FIXED_PARAMETERS['udpipe_path'],
                             seq_length=FIXED_PARAMETERS['seq_length'], r=FIXED_PARAMETERS['s2_dim'],
                             cache_file=os.path.join(FIXED_PARAMETERS["log_path"], modname)+'.test_matched.cache')
test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"], udpipe_path=FIXED_PARAMETERS['udpipe_path'],
                                seq_length=FIXED_PARAMETERS['seq_length'], r=FIXED_PARAMETERS['s2_dim'],
                                cache_file=os.path.join(FIXED_PARAMETERS["log_path"], modname)+'.test_mismatched.cache')

dictpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".p"

train_features_file = os.path.join(FIXED_PARAMETERS["log_path"], modname)+'.train.features.npy'
dev_features_file = os.path.join(FIXED_PARAMETERS["log_path"], modname)+'.dev.features.npy'

if not os.path.isfile(dictpath):
    print "No dictionary found!"
    exit(1)

else:
    logger.log("Loading dictionary from %s" % dictpath)
    word_indices = pickle.load(open(dictpath, "rb"))
    logger.log("Padding and indexifying sentences")
    sentences_to_padded_index_sequences(word_indices, [training_mnli, dev_matched, test_matched, test_mismatched])

loaded_embeddings = load_embedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)


class XGBClassifierModel:
    def __init__(self):
        # Define hyperparameters
        self.model_type = FIXED_PARAMETERS["model_type"]
        self.learning_rate = FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.alpha = FIXED_PARAMETERS["alpha"]
        self.udpipe_path = FIXED_PARAMETERS['udpipe_path']
        FIXED_PARAMETERS['embeddings'] = loaded_embeddings

        logger.log("Building model from %s.py" % model)
        self.model = MyModel(**FIXED_PARAMETERS)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(
            self.model.total_cost)

        # tf things: initialize variables and create placeholder for session
        logger.log("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

        best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver.restore(self.sess, best_path)
        logger.log("Model restored from file: %s" % best_path)

        self.clf = None

    def get_minibatch(self, dataset, start_index, end_index):
        if end_index > len(dataset):
            end_index = len(dataset)
        indices = range(start_index, end_index)
        premise_vectors = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in indices])
        hypothesis_vectors = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in indices])
        labels = [dataset[i]['label'] for i in indices]

        prem_dep = None
        hypo_dep = None

        if self.udpipe_path:
            prem_dep = np.stack([dataset[i]['prem_dep'] for i in indices])
            hypo_dep = np.stack([dataset[i]['hypo_dep'] for i in indices])

        return premise_vectors, hypothesis_vectors, labels, prem_dep, hypo_dep

    def train(self, train_set, dev_set):
        logger.log('Get features from training set')
        if os.path.exists(train_features_file):
            train_features = np.load(train_features_file)
            _, _, train_labels, _, _ = self.get_minibatch(train_set, 0, len(train_set))
        else:
            train_features = None
            train_labels = []
            total_batch = int(len(train_set) - 1)/self.batch_size + 1
            for i in tqdm(range(total_batch)):
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
                    minibatch_prem_dep, minibatch_hypo_dep = \
                    self.get_minibatch(train_set, i * self.batch_size, (i+1) * self.batch_size)
                feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                             self.model.hypothesis_x: minibatch_hypothesis_vectors,
                             self.model.y: minibatch_labels,
                             self.model.keep_rate_ph: 1.0}
                if 'dep_avg' in self.model_type:
                    feed_dict[self.model.prem_dep] = minibatch_prem_dep
                    feed_dict[self.model.hypo_dep] = minibatch_hypo_dep
                minibatch_features = self.sess.run([self.model.features], feed_dict)
                train_features = minibatch_features[0] if train_features is None \
                    else np.concatenate((train_features, minibatch_features[0]))
                train_labels += minibatch_labels

            np.save(train_features_file, train_features)

        logger.log('Get features from dev set')
        if os.path.exists(dev_features_file):
            dev_features = np.load(dev_features_file)
            _, _, dev_labels, _, _ = self.get_minibatch(dev_set, 0, len(dev_set))
        else:
            dev_features = None
            dev_labels = []
            total_batch = int(len(dev_set) - 1)/self.batch_size + 1
            for i in tqdm(range(total_batch)):
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
                    minibatch_prem_dep, minibatch_hypo_dep = \
                    self.get_minibatch(dev_set, i * self.batch_size, (i+1) * self.batch_size)
                feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                             self.model.hypothesis_x: minibatch_hypothesis_vectors,
                             self.model.y: minibatch_labels,
                             self.model.keep_rate_ph: 1.0}
                if 'dep_avg' in self.model_type:
                    feed_dict[self.model.prem_dep] = minibatch_prem_dep
                    feed_dict[self.model.hypo_dep] = minibatch_hypo_dep
                minibatch_features = self.sess.run([self.model.features], feed_dict)
                dev_features = minibatch_features[0] if dev_features is None \
                    else np.concatenate((dev_features, minibatch_features[0]))
                dev_labels += minibatch_labels

            np.save(dev_features_file, dev_features)

        tuned_parameters = {'max_depth': [4, 6, 8], 'n_estimators': [100, 200]}

        best_score = 0.
        best_params = []
        for g in ParameterGrid(tuned_parameters):
            clf = XGBClassifier(nthread=24)
            clf.set_params(**g)
            clf.fit(train_features, train_labels)
            score = clf.score(dev_features, dev_labels)
            logger.log('%s: %f' % (str(g), score))
            if best_score < score:
                best_score = score
                best_params = g
                self.clf = clf

        logger.log('Best score: %s %f' % (str(best_params), best_score))

    def classify(self, examples):
        # This classifies a list of examples
        features = None
        total_batch = int(len(examples) - 1)/self.batch_size + 1
        for i in tqdm(range(total_batch)):
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
                minibatch_prem_dep, minibatch_hypo_dep = \
                self.get_minibatch(examples, i * self.batch_size, (i+1) * self.batch_size)
            feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                         self.model.hypothesis_x: minibatch_hypothesis_vectors,
                         self.model.y: minibatch_labels,
                         self.model.keep_rate_ph: 1.0}
            if 'dep_avg' in self.model_type:
                feed_dict[self.model.prem_dep] = minibatch_prem_dep
                feed_dict[self.model.hypo_dep] = minibatch_hypo_dep
            minibatch_features = self.sess.run([self.model.features], feed_dict)
            features = minibatch_features[0] if features is None else np.concatenate((features, minibatch_features[0]))

        return self.clf.predict(features)


classifier = XGBClassifierModel()
classifier.train(training_mnli, dev_matched)

"""
Get CSVs of predictions.
"""

logger.log("Creating CSV of predicitons on matched test set: %s" % (modname + "_matched_predictions.csv"))
predictions_kaggle(classifier.classify, test_matched, FIXED_PARAMETERS["batch_size"], modname+"_dev_matched")

logger.log("Creating CSV of predicitons on mismatched test set: %s" % (modname + "_mismatched_predictions.csv"))
predictions_kaggle(classifier.classify, test_mismatched, FIXED_PARAMETERS["batch_size"], modname+"_dev_mismatched")
