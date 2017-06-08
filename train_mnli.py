"""
Training script to train a model on MultiNLI and, optionally, on SNLI data as well.
The "alpha" hyperparamaters set in paramaters.py determines if SNLI data is used in training. If alpha = 0, no SNLI data
is used in training. If alpha > 0, then down-sampled SNLI data is used in training.
"""
import os
import importlib
import pickle

import tensorflow as tf
import random
from utils import logger
import utils.parameters as params
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
logger.log("FIXED_PARAMETERS\n%s" % FIXED_PARAMETERS)

logger.log("Loading data")
training_snli = load_nli_data(FIXED_PARAMETERS["training_snli"], snli=True)
dev_snli = load_nli_data(FIXED_PARAMETERS["dev_snli"], snli=True)
test_snli = load_nli_data(FIXED_PARAMETERS["test_snli"], snli=True)

training_mnli = load_nli_data(FIXED_PARAMETERS["training_mnli"])
dev_matched = load_nli_data(FIXED_PARAMETERS["dev_matched"])
dev_mismatched = load_nli_data(FIXED_PARAMETERS["dev_mismatched"])
test_matched = load_nli_data(FIXED_PARAMETERS["test_matched"])
test_mismatched = load_nli_data(FIXED_PARAMETERS["test_mismatched"])

if 'temp.jsonl' in FIXED_PARAMETERS["test_matched"]:
    # Removing temporary empty file that was created in parameters.py
    os.remove(FIXED_PARAMETERS["test_matched"])
    logger.log("Created and removed empty file called temp.jsonl since test set is not available.")

dictpath = os.path.join(FIXED_PARAMETERS["log_path"], modname) + ".p"

if not os.path.isfile(dictpath):
    logger.log("Building dictionary")
    if FIXED_PARAMETERS["alpha"] == 0:
        word_indices = build_dictionary([training_mnli])
    else:
        word_indices = build_dictionary([training_mnli, training_snli])

    logger.log("Padding and indexifying sentences")
    sentences_to_padded_index_sequences(word_indices,
                                        [training_mnli, training_snli, dev_matched, dev_mismatched, dev_snli, test_snli,
                                         test_matched, test_mismatched])
    pickle.dump(word_indices, open(dictpath, "wb"))

else:
    logger.log("Loading dictionary from %s" % dictpath)
    word_indices = pickle.load(open(dictpath, "rb"))
    logger.log("Padding and indexifying sentences")
    sentences_to_padded_index_sequences(word_indices,
                                        [training_mnli, training_snli, dev_matched, dev_mismatched, dev_snli, test_snli,
                                         test_matched, test_mismatched])

logger.log("Loading embeddings")
loaded_embeddings = load_embedding_rand(FIXED_PARAMETERS["embedding_data_path"], word_indices)


class ModelClassifier:
    def __init__(self, seq_length):
        # Define hyperparameters
        self.learning_rate = FIXED_PARAMETERS["learning_rate"]
        self.display_epoch_freq = 1
        self.display_step_freq = 50
        self.embedding_dim = FIXED_PARAMETERS["word_embedding_dim"]
        self.dim = FIXED_PARAMETERS["hidden_embedding_dim"]
        self.batch_size = FIXED_PARAMETERS["batch_size"]
        self.emb_train = FIXED_PARAMETERS["emb_train"]
        self.keep_rate = FIXED_PARAMETERS["keep_rate"]
        self.sequence_length = FIXED_PARAMETERS["seq_length"]
        self.alpha = FIXED_PARAMETERS["alpha"]

        logger.log("Building model from %s.py" % model)
        self.model = MyModel(seq_length=self.sequence_length, emb_dim=self.embedding_dim, hidden_dim=self.dim,
                             embeddings=loaded_embeddings, emb_train=self.emb_train)

        # Perform gradient descent with Adam
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999).minimize(
            self.model.total_cost)

        # Boolean stating that training has not been completed,
        self.completed = False

        # tf things: initialize variables and create placeholder for session
        logger.log("Initializing variables")
        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

    def get_minibatch(self, dataset, start_index, end_index):
        indices = range(start_index, end_index)
        premise_vectors = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in indices])
        hypothesis_vectors = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in indices])
        genres = [dataset[i]['genre'] for i in indices]
        labels = [dataset[i]['label'] for i in indices]
        return premise_vectors, hypothesis_vectors, labels, genres

    def train(self, train_mnli, train_snli, dev_mat, dev_mismat, dev_snli):
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.step = 0
        self.epoch = 0
        self.best_dev_mat = 0.
        self.best_mtrain_acc = 0.
        self.last_train_acc = [.001, .001, .001, .001, .001]
        self.best_step = 0

        # Restore most recent checkpoint if it exists.
        # Also restore values for best dev-set accuracy and best training-set accuracy
        ckpt_file = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        if os.path.isfile(ckpt_file + ".meta"):
            if os.path.isfile(ckpt_file + "_best.meta"):
                self.saver.restore(self.sess, (ckpt_file + "_best"))
                self.best_dev_mat, dev_cost_mat = evaluate_classifier(self.classify, dev_mat, self.batch_size)
                best_dev_mismat, dev_cost_mismat = evaluate_classifier(self.classify, dev_mismat, self.batch_size)
                best_dev_snli, dev_cost_snli = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                self.best_mtrain_acc, mtrain_cost = evaluate_classifier(self.classify, train_mnli[0:5000],
                                                                        self.batch_size)
                if self.alpha != 0.:
                    self.best_strain_acc, strain_cost = evaluate_classifier(self.classify, train_snli[0:5000],
                                                                            self.batch_size)
                    logger.log(
                        "Restored best matched-dev acc: %f\nRestored best mismatched-dev acc: %f\nRestored best "
                        "SNLI-dev acc: %f\nRestored best MulitNLI train acc: %f\nRestored best SNLI train acc: %f" %
                        (self.best_dev_mat, best_dev_mismat, best_dev_snli, self.best_mtrain_acc, self.best_strain_acc))
                else:
                    logger.log(
                        "Restored best matched-dev acc: %f\nRestored best mismatched-dev acc: %f\nRestored best "
                        "SNLI-dev acc: %f\nRestored best MulitNLI train acc: %f" % ( 
                            self.best_dev_mat, best_dev_mismat, best_dev_snli, self.best_mtrain_acc))

            self.saver.restore(self.sess, ckpt_file)
            logger.log("Model restored from file: %s" % ckpt_file)

        # Combine MultiNLI and SNLI data. Alpha has a default value of 0, if we want to use SNLI data, it must be
        # passed as an argument.
        beta = int(self.alpha * len(train_snli))

        # Training cycle
        logger.log("Training...")
        logger.log("Model will use %s percent of SNLI data during training" % (self.alpha * 100))

        while True:
            training_data = train_mnli + random.sample(train_snli, beta)
            random.shuffle(training_data)
            avg_cost = 0.
            total_batch = int(len(training_data) / self.batch_size)

            # Loop over all batches in epoch
            for i in range(total_batch):
                # Assemble a minibatch of the next B examples
                minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres = \
                    self.get_minibatch(training_data, self.batch_size * i, self.batch_size * (i + 1))

                # Run the optimizer to take a gradient step, and also fetch the value of the
                # cost function for logging
                feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                             self.model.hypothesis_x: minibatch_hypothesis_vectors,
                             self.model.y: minibatch_labels,
                             self.model.keep_rate_ph: self.keep_rate}
                _, c = self.sess.run([self.optimizer, self.model.total_cost], feed_dict)

                # Compute average loss
                avg_cost += c / (total_batch * self.batch_size)

                if self.step % self.display_step_freq == 0:
                    logger.log("Epoch %i, step %i\tAvg. train cost: %f" % (
                        self.epoch, self.step, avg_cost * total_batch / i))
                    # if self.alpha != 0.:
                    #     strain_acc, strain_cost = evaluate_classifier(self.classify, train_snli[0:5000],
                    #                                                   self.batch_size)
                    # dev_acc_snli, dev_cost_snli = evaluate_classifier(self.classify, dev_snli, self.batch_size)
                    #     logger.log(
                    #         "Epoch %i, step %i\tDev-matched acc: %f\tDev-mismatched acc: %f\tDev-SNLI acc: %f\t"
                    #         "MultiNLI train acc: %f\tSNLI train acc: %f" % (
                    #             self.step, dev_acc_mat, dev_acc_mismat, dev_acc_snli, mtrain_acc, strain_acc))
                    #     logger.log(
                    #         "Epoch %i, step %i\tDev-matched cost: %f\tDev-mismatched cost: %f\tDev-SNLI cost: %f\t"
                    #         "MultiNLI train cost: %f\tSNLI train cost: %f" % (
                    #             self.step, dev_cost_mat, dev_cost_mismat, dev_cost_snli, mtrain_cost, strain_cost))
                    # else:

                if self.step % 500 == 0:
                    mtrain_acc, mtrain_cost = evaluate_classifier(self.classify, train_mnli[0:5000], self.batch_size)
                    dev_acc_mat, dev_cost_mat = evaluate_classifier(self.classify, dev_mat, self.batch_size)
                    dev_acc_mismat, dev_cost_mismat = evaluate_classifier(self.classify, dev_mismat, self.batch_size)

                    logger.log("Epoch %i, step %i\tMultiNLI train acc: %f\tMultiNLI train cost: %f" % (
                        self.epoch, self.step, mtrain_acc, mtrain_cost))
                    logger.log("Epoch %i, step %i\tDev-matched acc: %f\tDev-mismatched acc: %f\tDev-matched cost: %f"
                               "\tDev-mismatched cost: %f" % (self.epoch, self.step, dev_acc_mat,
                                                              dev_acc_mismat, dev_cost_mat, dev_cost_mismat))

                    self.saver.save(self.sess, ckpt_file)
                    best_test = 100 * (1 - self.best_dev_mat / dev_acc_mat)
                    if best_test > 0.04:
                        self.saver.save(self.sess, ckpt_file + "_best")
                        self.best_dev_mat = dev_acc_mat
                        self.best_mtrain_acc = mtrain_acc
                        # if self.alpha != 0.:
                        #     self.best_strain_acc = strain_acc
                        self.best_step = self.step
                        logger.log("==== Checkpointing with new best matched-dev accuracy: %f" % self.best_dev_mat)

                self.step += 1

            # Display some statistics about the epoch
            if self.epoch % self.display_epoch_freq == 0:
                logger.log("======= Epoch: %i\tAvg. Cost: %f" % (self.epoch + 1, avg_cost))

            self.epoch += 1
            self.last_train_acc[(self.epoch % 5) - 1] = mtrain_acc

            # Early stopping
            progress = 1000 * (sum(self.last_train_acc) / (5 * min(self.last_train_acc)) - 1)

            if (progress < 0.1) or (self.step > self.best_step + 30000):
                logger.log("Best matched-dev accuracy: %s" % self.best_dev_mat)
                logger.log("MultiNLI Train accuracy: %s" % self.best_mtrain_acc)
                self.completed = True
                break

    def classify(self, examples):
        # This classifies a list of examples
        if test or self.completed:
            best_path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
            self.sess = tf.Session()
            self.sess.run(self.init)
            self.saver.restore(self.sess, best_path)
            logger.log("Model restored from file: %s" % best_path)

        total_batch = int(len(examples) / self.batch_size)
        logits = np.empty(3)
        cost = 0
        genres = []
        for i in range(total_batch):
            minibatch_premise_vectors, minibatch_hypothesis_vectors, minibatch_labels, minibatch_genres = \
                self.get_minibatch(examples, self.batch_size * i, self.batch_size * (i + 1))
            feed_dict = {self.model.premise_x: minibatch_premise_vectors,
                         self.model.hypothesis_x: minibatch_hypothesis_vectors,
                         self.model.y: minibatch_labels,
                         self.model.keep_rate_ph: 1.0}
            genres += minibatch_genres
            logit, cost = self.sess.run([self.model.logits, self.model.total_cost], feed_dict)
            logits = np.vstack([logits, logit])

        return genres, np.argmax(logits[1:], axis=1), cost

    def restore(self, best=True):
        if best:
            path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt_best"
        else:
            path = os.path.join(FIXED_PARAMETERS["ckpt_path"], modname) + ".ckpt"
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver.restore(self.sess, path)
        logger.log("Model restored from file: %s" % path)


classifier = ModelClassifier(FIXED_PARAMETERS["seq_length"])

"""
Either train the model and then run it on the test-sets or
load the best checkpoint and get accuracy on the test set. Default setting is to train the model.
"""

test = params.train_or_test()

# While test-set isn't released, use dev-sets for testing
test_matched = dev_matched
test_mismatched = dev_mismatched

if not test:
    classifier.train(training_mnli, training_snli, dev_matched, dev_mismatched, dev_snli)
    logger.log("Acc on matched multiNLI dev-set: %s" %
               (evaluate_classifier(classifier.classify, test_matched, FIXED_PARAMETERS["batch_size"]))[0])
    logger.log("Acc on mismatched multiNLI dev-set: %s" %
               (evaluate_classifier(classifier.classify, test_mismatched, FIXED_PARAMETERS["batch_size"]))[0])
    logger.log("Acc on SNLI test-set: %s" %
               (evaluate_classifier(classifier.classify, test_snli, FIXED_PARAMETERS["batch_size"]))[0])
else:
    results = evaluate_final(classifier.restore, classifier.classify, [test_matched, test_mismatched, test_snli],
                             FIXED_PARAMETERS["batch_size"])
    logger.log("Acc on multiNLI matched dev-set: %s" % (results[0]))
    logger.log("Acc on multiNLI mismatched dev-set: %s" % (results[1]))
    logger.log("Acc on SNLI test set: %s" % (results[2]))

    # Results by genre,
    logger.log("Acc on matched genre dev-sets: %s" % (
        evaluate_classifier_genre(classifier.classify, test_matched, FIXED_PARAMETERS["batch_size"])[0]))
    logger.log("Acc on mismatched genres dev-sets: %s" % (
        evaluate_classifier_genre(classifier.classify, test_mismatched, FIXED_PARAMETERS["batch_size"])[0]))
