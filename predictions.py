"""
Script to generate a CSV file of predictions on the test data.
"""

import tensorflow as tf
import os
import importlib
from utils import logger
import utils.parameters as params
from utils.data_processing import *
from utils.evaluate import *
import pickle

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
test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"], udpipe_path=FIXED_PARAMETERS['udpipe_path'],
                             seq_length=FIXED_PARAMETERS['seq_length'], r=FIXED_PARAMETERS['s2_dim'],
                             cache_file=os.path.join(FIXED_PARAMETERS["log_path"], modname)+'.test_matched.cache')
test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"], udpipe_path=FIXED_PARAMETERS['udpipe_path'],
                                seq_length=FIXED_PARAMETERS['seq_length'], r=FIXED_PARAMETERS['s2_dim'],
                                cache_file=os.path.join(FIXED_PARAMETERS["log_path"], modname)+'.test_mismatched.cache')

dictpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".p"

if not os.path.isfile(dictpath): 
    print "No dictionary found!"
    exit(1)

else:
    logger.log("Loading dictionary from %s" % dictpath)
    word_indices = pickle.load(open(dictpath, "rb"))
    logger.log("Padding and indexifying sentences")
    sentences_to_padded_index_sequences(word_indices, [test_matched, test_mismatched])

loaded_embeddings = load_embedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)


class ModelClassifier:
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

    def get_minibatch(self, dataset, start_index, end_index):
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

    def classify(self, examples):
        # This classifies a list of examples
        best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver.restore(self.sess, best_path)
        logger.log("Model restored from file: %s" % best_path)

        logits = np.empty(3)
        minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, \
            minibatch_prem_dep, minibatch_hypo_dep = \
            self.get_minibatch(examples, 0, len(examples))
        feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                     self.model.hypothesis_x: minibatch_hypothesis_vectors,
                     self.model.y: minibatch_labels,
                     self.model.keep_rate_ph: 1.0}
        if 'dep_avg' in self.model_type:
            feed_dict[self.model.prem_dep] = minibatch_prem_dep
            feed_dict[self.model.hypo_dep] = minibatch_hypo_dep
        logit, cost = self.sess.run([self.model.logits, self.model.total_cost], feed_dict)
        logits = np.vstack([logits, logit])

        return np.argmax(logits[1:], axis=1)


classifier = ModelClassifier()

"""
Get CSVs of predictions.
"""

logger.log("Creating CSV of predicitons on matched test set: %s" % (modname + "_matched_predictions.csv"))
predictions_kaggle(classifier.classify, test_matched, FIXED_PARAMETERS["batch_size"], modname+"_dev_matched")

logger.log("Creating CSV of predicitons on mismatched test set: %s" % (modname + "_mismatched_predictions.csv"))
predictions_kaggle(classifier.classify, test_mismatched, FIXED_PARAMETERS["batch_size"], modname+"_dev_mismatched")
