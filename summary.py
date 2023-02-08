import tensorflow as tf
from parser import args
import sys

class Summary():

    def __init__(self, experiment=None):
        self.experiment = arg.exp if experiment is None else experiment
        self.writer = tf.summary.create_file_writer('runs/' + self.experiment + '/')

        with self.writer.as_default():
            tf.summary.text('Arguments', sys.argv, step=0)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.max_train_accuracy = 0.0

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.max_test_accuracy = 0.0


        self.all_train = [self.train_loss, self.train_accuracy]
        self.all_test = [self.test_loss, self.test_accuracy]

    def wipe(self, eval=False):
        if(eval):
            for metric in self.all_test:
                metric.reset_states()
        else:
            for metric in self.all_train:
                metric.reset_states()

    def add_train(self, loss, label, pred):
        self.train_loss(loss)
        self.train_accuracy(label, pred)

    def add_test(self, loss, label, pred):
        self.test_loss(loss)
        self.test_accuracy(label, pred)

    def step(self, loss, label, pred, eval=False):
        if(not eval):
            self.add_train(loss, label, pred)
        else:
            self.add_test(loss, label, pred)

    def write(self, steps, eval=False):
        if(not eval):
            if(self.max_train_accuracy < self.train_accuracy.result() * 100):
                self.max_train_accuracy = self.train_accuracy.result() * 100

            tf.summary.scalar('train_loss/loss', self.train_loss.result(), steps)
            tf.summary.scalar('train_accuracy/accuracy', self.train_accuracy.result() * 100, steps)
        else:
            if(self.max_test_accuracy < self.test_accuracy.result() * 100):
                self.max_test_accuracy = self.test_accuracy.result() * 100

            tf.summary.scalar('test_loss/loss', self.test_loss.result(), steps)
            tf.summary.scalar('test_accuracy/accuracy', self.test_accuracy.result() * 100, steps)

        self.writer.flush()
        self.wipe(eval=eval)
