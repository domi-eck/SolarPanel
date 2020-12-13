from argparse import ArgumentParser
from pathlib import Path
import os
from matplotlib import pyplot as plt
import save_helper as sh

import tensorflow as tf

import data
import train
import model
import evaluation


# general settings
EVAL_MEASURES   = ['ClasswiseAccuracy', 'ClasswiseRecall', 'ClasswisePrecision', 'ClasswiseF1']
CLASSNAMES      = ['crack', 'inactive']
NUM_CLASSES     = len(CLASSNAMES)
DATASET_TRAIN   = Path('.') / 'data' / 'train.csv'

# training related settings
# TODO adapt these to your needs and find suitable hyperparameter settings
MODEL           = 'alexnet'
BATCH_SIZE      = 10
LEARNING_RATE   = 0.0001
SAVE_DIR        = Path('/proj/ciptmp/iz11ocuc/out_alexnet_smaller_learning_rate')
STOP_PATIENCE   = 10            # wait for the loss to increase for this many epochs until training stops
TRAIN_PART      = 0.75
SHUFFLE         = True
AUGMENT         = True


if __name__ == '__main__':

    ds = data.Dataset(DATASET_TRAIN, CLASSNAMES, BATCH_SIZE, SHUFFLE, AUGMENT)
    ds_train, ds_validation = ds.split_train_validation(TRAIN_PART)
    print('Dataset split into {:d} training and {:d} validation samples'.format(len(ds_train), len(ds_validation)))

    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):

        # Note: We need to use placeholders for inputs and outputs. Otherwise,
        # the batch size would be fixed and we could not use the trained model with
        # a different batch size. In addition, the names of these tensors must be "inputs"
        # and "labels" such that we can find them on the evaluation server. DO NOT CHANGE THIS!
        x = tf.placeholder(tf.float32, [None] + [224,224] + [1], 'inputs')
        labels = tf.placeholder(tf.float32, [None] + [NUM_CLASSES], 'labels')
        
        if MODEL == 'alexnet':
            prediction_logits = model.alexnet(x, len(CLASSNAMES))
        if MODEL == 'resnet':
            prediction_logits = model.resnet(x, len(CLASSNAMES))

        # apply loss
        # TODO : Implement suitable loss function for multi-label problem
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=prediction_logits)
        mse_loss = tf.losses.mean_squared_error(labels=labels, predictions=prediction_logits)


        # convert into binary (boolean) predictions
        prediction_sigmoid = tf.nn.sigmoid(prediction_logits)
        prediction_int = tf.round(prediction_sigmoid)
        prediction_bin = tf.cast(prediction_int,tf.bool)


        
        # Note: The name of the predictions tensor needs to be "predictions" such that
        # we can identify it when loading the model on the evaluation server.
        # We use tf.identity to give it a fixed name.
        prediction = tf.identity(prediction_bin, name = 'predictions')

    # Note: this is required to update batchnorm during training..
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        ev = evaluation.create_evaluation(EVAL_MEASURES, CLASSNAMES)
        trainer = train.Trainer(loss, prediction, optimizer, ds_train, ds_validation, STOP_PATIENCE, ev, x, labels, prediction_int, prediction_sigmoid, prediction_logits)


    # check if output directory does not exist
    if  SAVE_DIR.exists():
        print('directory {} must not exist..'.format(str(SAVE_DIR)))
        exit(1)


    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())

        # train
        print('Run training loop')
        trainer.run(sess)

        # save
        print('Save model')
        sh.simple_save(sess, str(SAVE_DIR), inputs = {'x': x}, outputs = {'y': prediction})

        # create zip file for submission
        print('Create zip file for submission')
        sh.zip_dir(SAVE_DIR, SAVE_DIR / 'model.zip')


