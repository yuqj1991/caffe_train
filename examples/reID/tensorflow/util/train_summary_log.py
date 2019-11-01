import tensorflow as tf
import os


# add some train log, like loss, accuracy
#

class train_log(object):
    def __int__(self, log_dir, loss, sess=None, accuracy=None):
        self.log = log_dir
        self.loss = loss
        self.accuracy = accuracy
        self.sess = sess
        try:
            self.image_summary = tf.image_summary
            self.scalar_summary = tf.scalar_summary
            self.histogram_summary = tf.histogram_summary
            self.merge_summary = tf.merge_summary
            self.SummaryWriter = tf.train.SummaryWriter
        except:
            self.image_summary = tf.summary.image
            self.scalar_summary = tf.summary.scalar
            self.histogram_summary = tf.summary.histogram
            self.merge_summary = tf.summary.merge
            self.SummaryWriter = tf.summary.FileWriter

    def add_loss_summaries(self, collection_name):
        """
        Add summaries for losses.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        Args:
          collection_name: Total loss collection name.
        Returns:
          loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection(key=collection_name)  # losses is a list value
        loss_averages_op = loss_averages.apply(losses + [self.loss])  # self.loss is also a list value
        # Attach a scalar summmary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [self.loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            self.scalar_summary(l.op.name + ' (raw)', l)
            self.scalar_summary(l.op.name, loss_averages.average(l))
        return loss_averages_op

    def add_accuracy_summaries(self, grads, log_histograms=True):
        """
        add summaries for accuracy
        :return:
        accuracy_op
        """
        self.scalar_summary('accuracy', self.accuracy)
        # Add histograms for trainable variables.
        if log_histograms:
            for var in tf.trainable_variables():
                self.histogram_summary(var.op.name, var)

        # Add histograms for gradients.
        if log_histograms:
            for grad, var in grads:
                if grad is not None:
                    self.histogram_summary(var.op.name + '/gradients', grad)

    def save_log_file(self, val_list):
        merged = self.merge_summary(inputs=val_list)
        writer = self.SummaryWriter(self.log, self.sess.graph)
        return merged, writer

