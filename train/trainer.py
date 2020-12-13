import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as sm
import evaluation.measures as em

class Trainer:

    def __init__(self, loss, predictions, optimizer, ds_train, ds_validation, stop_patience, evaluation,
                 inputs, labels, prediction_int, prediction_sigmoid, prediction_logits):
        '''
            Initialize the trainer

            Args:
                loss        	an operation that computes the loss
                predictions     an operation that computes the predictions for the current
                optimizer       optimizer to use
                ds_train        instance of Dataset that holds the training data
                ds_validation   instance of Dataset that holds the validation data
                stop_patience   the training stops if the validation loss does not decrease for this number of epochs
                evaluation      instance of Evaluation
                inputs          placeholder for model inputs
                labels          placeholder for model labels
        '''

        self._train_op = optimizer.minimize(loss)

        self._loss = loss
        self._predictions = predictions
        self._ds_train = ds_train
        self._ds_validation = ds_validation
        self._stop_patience = stop_patience
        self._evaluation = evaluation
        self._validation_losses = []
        self._model_inputs = inputs
        self._model_labels = labels
        self._count_epoch = 0
        self._prediction_int = prediction_int
        self._prediction_sigmoid = prediction_sigmoid
        self._prediction_logits = prediction_logits
        self._train_count_epoch = 0
        self._loss_array = np.zeros(40)
        self._epsilon = 0.01


        with tf.variable_scope('model', reuse = True):
            self._model_is_training = tf.get_variable('is_training', dtype = tf.bool)

    def _train_epoch(self, sess):
        '''
            trains for one epoch and prints the mean training loss to the commandline

            args:
                sess    the tensorflow session that should be used
        '''
        self._train_count_epoch += 1
        print("\n")
        print("######NEW_EPOCH ", self._train_count_epoch, "########################################")
        print("Training_______________________________________________")
        print("epoch: ", self._train_count_epoch)
        self.mean_loss = np.array([])
        predict_array = np.empty((0,2), dtype=np.float)
        label_array= np.empty((0,2), dtype=np.float)

        for inputs, labels in self._ds_train:
            train, loss, predict, pred_sigmoid, pred_logit = sess.run([ self._train_op,self._loss, self._predictions,
                                                                        self._prediction_sigmoid, self._prediction_logits],
            feed_dict={self._model_inputs: inputs, self._model_labels: labels, self._model_is_training : True})

            self.mean_loss = np.append(self.mean_loss, loss)
            predict_array = np.append(predict_array, predict, axis=0)
            label_array = np.append(label_array, labels, axis=0)
            #print("pred_sigmoid, pred_logit, predict, labels")
            #test = zip(np.round(pred_sigmoid[0:5], 2), np.round(pred_logit[0:5], 2), predict[0:5], labels[0:5])
            # for i,j,z, y in test:
            #     print(i,j,z, y)
            # print("\n")


        f1_score = sm.f1_score(label_array, predict_array, average="micro")
        self._evaluation.add_batch(predict_array, label_array)
        # print("\n")
        print('mean_loss: ', np.mean(self.mean_loss))
        print(' ')
        print( "evaluation :")
        self._evaluation.flush()
        print('\n')
        pass

    def _valid_step(self, sess):
        '''
            run the validation and print evalution + mean validation loss to the commandline

            args:
                sess    the tensorflow session that should be used


        '''
        print('\033[1m')
        print("Validation_______________________________________________")
        predict_array = np.empty((0,2), dtype=np.float)
        label_array= np.empty((0,2), dtype=np.float)
        self.mean_loss = np.array([])
        for inputs, labels in self._ds_validation:
            loss, predict, predict_int = sess.run([self._loss, self._predictions, self._prediction_int],
                feed_dict={self._model_inputs: inputs, self._model_labels: labels, self._model_is_training : False})
            predict_array = np.append(predict_array, predict, axis=0)
            label_array = np.append(label_array, labels, axis=0)

            self.mean_loss = np.append(self.mean_loss, loss)

        self._evaluation.add_batch(predict_array, label_array)
        f1_score = sm.f1_score(label_array, predict_array, average="micro")


        self._loss_array[self._count_epoch%20 +20] = loss
        self._count_epoch +=1
        print('mean_loss: ', np.mean(self.mean_loss))
        print('')

        print('evaluations: ')
        self._evaluation.flush()
        print('\033[0m' )

        pass

    def _should_stop(self):
        '''
            determine if training should stop according to stop_patience
        '''
        if (self._count_epoch%20) == 0:
            former_mean = np.mean(self._loss_array[0:20])
            new_mean = np.mean(self._loss_array[20:40])


            print('\n')
            print("###### Check Stop Condition #################")
            print("mean1: ", former_mean)
            print(self._loss_array[0:20])
            print("mean2: ", new_mean)
            print(self._loss_array[20:40])

            self._loss_array[0:20] = self._loss_array[20:40]

            if not (self._count_epoch == 20):
                if np.isclose(former_mean, new_mean, self._epsilon) or (new_mean > former_mean):
                    print("stop Training")
                    return True


    def run(self, sess, num_epochs = -1):
        '''
            run the training until num_epochs exceeds or the validation loss did not decrease
            for stop_patience epochs

            args:
                sess        the tensorflow session that should be used
                num_epochs  limit to the number of epochs, -1 means not limit
        '''

        # initial validation step
        self._valid_step(sess)

        i = 0

        # training loop
        while i < num_epochs or num_epochs == -1:
            self._train_epoch(sess)
            self._valid_step(sess)
            i += 1

            if self._should_stop():
                break

