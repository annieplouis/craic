# Author: Annie Louis
# Seq2Seq model, based on the Tensorflow tutorial for MT
# We allow for teacher forcing in decoding or automatic decoding
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from six.moves import xrange  

import os.path
import numpy as np
import tensorflow as tf
import json


class Seq2SeqModel(object):

  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               encoder_size,
               decoder_size,
               size,
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               num_samples=512,
               forward_only=False,
               use_decoder_inputs=True,
               dtype=tf.float32):
    """
    :param encoder_size: max size of the encoder sequence
    :param decoder_size: max size of the decoder sequence
    :param size: number of hidden units
    :param num_samples: number of samples for sampled softmax
    :param forward_only: no backward pass, use for decoding during test time
    :param use_decoder_inputs: if True, then teacher forcing is applied during decoding 
         (i.e the oracle decoder outputs are fed into the current timestep)
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.encoder_size = encoder_size
    self.decoder_size = decoder_size
    self.num_layers = num_layers
    self.size = size
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)
    self.use_decoder_inputs = use_decoder_inputs
    self.keep_probability = tf.placeholder(tf.float32, name="keep_probability")

    output_projection = None
    softmax_loss_function = None
    # sampled softmax 
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
      output_projection = (w, b)

      def sampled_loss(logits, labels):
        labels = tf.reshape(labels, [-1, 1])
        # using 32bit floats to avoid numerical instabilities.
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(logits, tf.float32)
        return tf.cast(
            tf.nn.sampled_softmax_loss(
                local_w_t,
                local_b,
                local_inputs,
                labels,
                num_samples,
                self.target_vocab_size),
            dtype)
      softmax_loss_function = sampled_loss

    def lstm_cell():
      return tf.nn.rnn_cell.BasicLSTMCell(size)
    def attn_cell():
      return tf.nn.rnn_cell.DropoutWrapper(lstm_cell(), output_keep_prob=self.keep_probability)

    cell = tf.nn.rnn_cell.MultiRNNCell([attn_cell() for _ in range(num_layers)])

    def seq2seq_f(encoder_inputs, decoder_inputs, feed_prev):
      return tf.nn.seq2seq.embedding_attention_seq2seq(
          encoder_inputs,
          decoder_inputs,
          cell,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_size=size,
          output_projection=output_projection,
          feed_previous=feed_prev,
          dtype=dtype)

    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(self.encoder_size): 
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(self.decoder_size + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    buckets = [(self.encoder_size, self.decoder_size)]

    # Training outputs and losses.
    if forward_only:
      self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, not self.use_decoder_inputs),
          softmax_loss_function=softmax_loss_function)
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [
              tf.matmul(output, output_projection[0]) + output_projection[1]
              for output in self.outputs[b]
          ]
    else:
      self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, not self.use_decoder_inputs),
          softmax_loss_function=softmax_loss_function)

    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      gradients = tf.gradients(self.losses[0], params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.gradient_norms.append(norm)
      self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.global_variables())
    self.initialize = tf.global_variables_initializer()
    

  def get_parameter_count(self):
    params = tf.trainable_variables()
    total_parameters = 0
    for variable in params:
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim.value
      total_parameters += variable_parameters
    return total_parameters


  def write_model_parameters(self, model_directory):
    parameters = {
      "num_layers": self.num_layers,
      "source_vocab_size": self.source_vocab_size,
      "target_vocab_size": self.target_vocab_size,
      "hidden_size": self.size,
      "encoder_size": self.encoder_size,
      "decoder_size": self.decoder_size,
      "total_parameters": self.get_parameter_count()
    }
    with open(self.parameters_file(model_directory), "w") as f:
      json.dump(parameters, f, indent=4)

  @staticmethod    
  def parameters_file(model_directory):
    return os.path.join(model_directory, "parameters.json")

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           forward_only, keep_prob):

    encoder_size = self.encoder_size
    decoder_size = self.decoder_size
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length in data not same as in model"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length in data not same as in model,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length in data not same as in model,"
                       " %d != %d." % (len(target_weights), decoder_size))

    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]
    input_feed[self.keep_probability] = keep_prob

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates,  
                     self.gradient_norms,  
                     self.losses] 
    else:
      output_feed = [self.losses[0]] 
      for l in xrange(decoder_size): 
        output_feed.append(self.outputs[0][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      gradient_norm = outputs[1]
      loss = outputs[2]
      return gradient_norm, loss, None 
    else:
      loss = outputs[0]
      output_logits = outputs[1:]
      return None, loss, output_logits

