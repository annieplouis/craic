# Author: Annie Louis
#
# Code to comment translation
# - Can be used to generate the comment for a given piece of code or 
# - calculate the perplexity of a given comment with respect to the code
#
# Seq2seq model based on the MT framework from tensorflow tutorials
#=======================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
from datetime import timedelta
import logging
from six.moves import xrange 

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops

import reader_seq2seq
import seq2seq_model


flags = tf.flags
flags.DEFINE_integer("encoder_size", 60, "maximum sequence length for encoder")
flags.DEFINE_integer("decoder_size", 60, "maximum sequence length for decoder")
flags.DEFINE_integer("start_epoch", 1, "the epoch to start if continuing off from previously trained model")

flags.DEFINE_integer("size", 300, "Size of each model layer.")
flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_integer("predict_batch_size", 64, "Batch size to use during prediction.")
flags.DEFINE_integer("from_vocab_size", 25000, "Code vocabulary size.")
flags.DEFINE_integer("to_vocab_size", 25000, "Comment vocabulary size.")

flags.DEFINE_string("data_dir", "/tmp", "Data directory")
flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
flags.DEFINE_string("from_train_data", None, "Training data.")
flags.DEFINE_string("to_train_data", None, "Training data.")
flags.DEFINE_string("from_dev_data", None, "Training data.")
flags.DEFINE_string("to_dev_data", None, "Training data.")
flags.DEFINE_string("from_test_data", None, "Test data.")
flags.DEFINE_string("to_test_data", None, "Test data.")
flags.DEFINE_string("test_predictions", None, "Test predictions")

flags.DEFINE_float("keep_probability", 0.35, "Keep probability")
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("learning_rate_decay_factor", 0.96, "Learning rate decays by this much.")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
flags.DEFINE_integer("max_epochs", 50, "Max number of epochs to run")
flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
flags.DEFINE_integer("improve_threshold", 5, "number of epochs wherein if the validation perplexity does not improve, we stop training")

flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
flags.DEFINE_boolean("predict", False, "Set to True for computing predictability.")
flags.DEFINE_boolean("test", False, "Set to True for computing test perplexity.")

FLAGS = flags.FLAGS
dtype = tf.float32

def get_gpu_config():
  gconfig = tf.ConfigProto()
  gconfig.gpu_options.per_process_gpu_memory_fraction = 0.3
  gconfig.allow_soft_placement = True
  return gconfig

class SEQ2SEQ_LM(object):

  def __init__(self, model):
    self._model = model

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           forward_only, keep_prob):
    return self._model.step(session, encoder_inputs, decoder_inputs, target_weights,
                            forward_only, keep_prob)
  
  def train(self, session, config, train_dataset, valid_dataset, exit_criteria, summary_dir):
    start_time = time.time()
    epoch = config.start_epoch
    previous_valid_log_ppx = []
    nglobal_steps = 0
    summary_writer = tf.summary.FileWriter(summary_dir, session.graph)
    learning_rate = config.learning_rate
    
    try:
      while True:
        print("EPOCH: %d Learning rate %0.3f" % (epoch, learning_rate)) 
        epoch_start_time = time.time()
        for num_empty_seq, ids_into_data, encoder_inputs, decoder_inputs, target_weights in train_dataset.batch_producer(config.batch_size):
          step_start_time = time.time()
          grad_norm, loss, _ = self._model.step(session, encoder_inputs, decoder_inputs, target_weights, False, config.keep_probability)
          nglobal_steps += 1

        #one epoch is over
        train_log_perplexity = self.test(session, config, train_dataset)
        train_perplexity = perplexity_from_log(train_log_perplexity)
        valid_log_perplexity = self.test(session, config, valid_dataset)
        valid_perplexity = perplexity_from_log(valid_log_perplexity)

        train_perplexity_summary = tf.Summary() 
        train_perplexity_summary.value.add(tag="train_log_ppx", simple_value=train_log_perplexity)
        train_perplexity_summary.value.add(tag="train_ppx", simple_value=train_perplexity)
        summary_writer.add_summary(train_perplexity_summary, nglobal_steps)

        valid_perplexity_summary = tf.Summary()
        valid_perplexity_summary.value.add(tag="valid_log_ppx", simple_value=valid_log_perplexity)
        valid_perplexity_summary.value.add(tag="valid_ppx", simple_value=valid_perplexity)
        summary_writer.add_summary(valid_perplexity_summary, nglobal_steps)

        checkpoint_path = os.path.join(summary_dir, "s2s.ckpt.epoch" + str(epoch))
        self._model.saver.save(session, checkpoint_path, global_step=self._model.global_step)

        epoch_time = (time.time() - epoch_start_time) * 1.0 / 60
        print ("END EPOCH %d global_steps %d learning_rate %.4f time(mins) %.4f train_perplexity %.2f valid_perplexity %.2f" % (epoch, nglobal_steps, learning_rate, epoch_time, train_perplexity, valid_perplexity))
        sys.stdout.flush()

        if epoch > exit_criteria.max_epochs:
          raise StopTrainingException()

        # Decrease learning rate if validation perplexity increases 
        if len(previous_valid_log_ppx) > 1 and valid_log_perplexity > previous_valid_log_ppx[-1]:
          session.run(self._model.learning_rate_decay_op)
        learning_rate = self._model.learning_rate.eval()
        
        # If validation perplexity has not improved over the last n epochs, stop training 
        if learning_rate == 0.0 or \
           (len(previous_valid_log_ppx) >= exit_criteria.improve_threshold \
            and valid_log_perplexity > max(previous_valid_log_ppx[-exit_criteria.improve_threshold:])):
          raise StopTrainingException()

        previous_valid_log_ppx.append(valid_log_perplexity)
        epoch += 1

    except (StopTrainingException, KeyboardInterrupt):
      print("Training complete...")
      return (epoch, train_perplexity, valid_perplexity)

  def test(self, session, config, test_dataset):
    test_log_prob, test_total_size = 0.0, 0.0
    #currently num_empty_seq is not used in any useful way. 
    for num_empty_seq, ids_into_data, encoder_inputs, decoder_inputs, target_weights in test_dataset.batch_producer(config.batch_size):
      _, test_loss, _ = self._model.step(session, encoder_inputs, decoder_inputs, target_weights, True, 1.0)
      log_test_step_perp = test_loss
      test_step_size = np.sum(sum(target_weights))
      log_test_step_prob = -1 * test_step_size * log_test_step_perp
      test_log_prob += log_test_step_prob
      test_total_size += test_step_size
    test_log_ppx = -1 * test_log_prob / test_total_size 
    return test_log_ppx

def perplexity_from_log(log_ppx):
  ppx = math.exp(float(log_ppx)) if log_ppx < 300 else float("inf") 
  return ppx

def create_model(session, forward_only, use_decoder_inputs, config, train_dir):
  model = seq2seq_model.Seq2SeqModel(config.svocab_size, config.tvocab_size, config.encoder_size, config.decoder_size, config.hidden_size, config.num_layers, config.max_grad_norm, config.batch_size, config.learning_rate, config.learning_rate_decay, forward_only=forward_only, use_decoder_inputs=use_decoder_inputs, dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    with tf.device('/cpu:0'):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return SEQ2SEQ_LM(model)

      
def do_test(test_dataset, config, train_dir):
  with tf.device('/gpu:0'):
    with tf.Session(config=get_gpu_config()) as sess:
      s2s_model = create_model(sess, True, True, config, train_dir)
      test_log_ppx = s2s_model.test(sess, config, test_dataset) 
      test_ppx = perplexity_from_log(test_log_ppx) 
      print("Test perplexity " + str(test_ppx))


def calculate_predictability(from_test_file, to_test_file, prediction_file, svocab, rev_svocab, tvocab, rev_tvocab, config, train_dir):
  fout = open(prediction_file, "w")
  done_examples = 0   
  with tf.device('/gpu:0'):
    with tf.Session(config=get_gpu_config()) as sess:
      s2s_model = create_model(sess, True, True, config, train_dir)
      with tf.gfile.GFile(from_test_file, mode="r") as source_file:
        with tf.gfile.GFile(to_test_file, mode="r") as target_file:
          source, target = source_file.readline(), target_file.readline()
          while source and target:
            current_set = [(source.strip(), target.strip())]
            test_dataset = reader_seq2seq._dataset_from_lines(current_set, svocab, tvocab, config.encoder_size, config.decoder_size)
            log_ppx = s2s_model.test(sess, config, test_dataset)
            ppx = perplexity_from_log(log_ppx)
            fout.write(str(ppx) + "\n")
            done_examples += 1
            if done_examples % 500 == 0:
              print(" done " + str(done_examples))
              sys.stdout.flush()
            source, target = source_file.readline(), target_file.readline()
  fout.close()


def decode(from_test_file, prediction_file, svocab, rev_svocab, tvocab, rev_tvocab, config, train_dir):
  fout = open(prediction_file, "w")
  done_examples = 0
  with tf.device('/gpu:0'):
    with tf.Session(config=get_gpu_config()) as sess:
      s2s_model = create_model(sess, True, False, config, train_dir)
      print("Processing..")
      with tf.gfile.GFile(from_test_file, mode="r") as source_file:
        source = source_file.readline()
        target = reader_seq2seq.END_SENT
        while source:
          current_set = [(source.strip(), target.strip())]
          test_dataset = reader_seq2seq._dataset_from_lines(current_set, svocab, tvocab, config.encoder_size, config.decoder_size)

          for num_empty, index_into_data, encoder_inputs, decoder_inputs, target_weights in test_dataset.batch_producer(1):
            _, _, output_logits = s2s_model.step(sess, encoder_inputs, decoder_inputs, target_weights, True, 1.0)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

            end_sent_id = reader_seq2seq.special_symbols[reader_seq2seq.END_SENT]
            if end_sent_id in outputs:
              outputs = outputs[:outputs.index(end_sent_id)]

            gen_sent = " ".join([tf.compat.as_str(rev_tvocab[output]) for output in outputs])
            fout.write(gen_sent.strip() + "\n")
            done_examples += 1
            if done_examples % 100 == 0:
              print(str(done_examples) + " ", end="")
          source = source_file.readline()
  print("\nDone!")
  fout.close()
                     

  
class StopTrainingException(Exception):
  pass

class ExitCriteria(object):
  def __init__(self, max_epochs, impr_thres):
    self.max_epochs = max_epochs
    self.improve_threshold = impr_thres

def options_error(msg):
  print("\n\nError!! exiting..")
  print("\t--> " + msg)
  sys.exit(2)


class Config(object):
  def __init__(self, svocab_size, tvocab_size, encoder_size, decoder_size, hsize, num_layers, grad_norm, batch_size, lr, lr_decay, kp, start_epoch, max_epoch, improv_thres):
    self.svocab_size = svocab_size
    self.tvocab_size = tvocab_size
    self.encoder_size = encoder_size
    self.decoder_size = decoder_size
    self.hidden_size = hsize
    self.num_layers = num_layers
    self.max_grad_norm = grad_norm
    self.batch_size = batch_size
    self.learning_rate = lr
    self.learning_rate_decay = lr_decay
    self.start_epoch = start_epoch
    self.keep_probability = kp
    self.max_epoch = max_epoch
    self.improve_threshold = improv_thres
    
def main(_):

  svocab_path = os.path.join(FLAGS.train_dir, "vocab%d.from" % FLAGS.from_vocab_size)
  tvocab_path = os.path.join(FLAGS.train_dir, "vocab%d.to" % FLAGS.to_vocab_size)

  config = Config(FLAGS.from_vocab_size, FLAGS.to_vocab_size, FLAGS.encoder_size, FLAGS.decoder_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.keep_probability, FLAGS.start_epoch, FLAGS.max_epochs, FLAGS.improve_threshold)
  exit_criteria = ExitCriteria(FLAGS.max_epochs, FLAGS.improve_threshold)
  
  if FLAGS.predict:
    if FLAGS.from_test_data and FLAGS.to_test_data:
      from_test_file = FLAGS.from_test_data
      to_test_file = FLAGS.to_test_data
      predictions_teacher_file = FLAGS.test_predictions + ".teach"
    else:
      options_error("Arg error: test files must be given for decode and predict")
    print("Initializing vocabularies...")
    svocab, rev_svocab = reader_seq2seq._read_vocab(svocab_path)
    tvocab, rev_tvocab = reader_seq2seq._read_vocab(tvocab_path)
    print("Calculating predictability...")
    config.batch_size = 1
    calculate_predictability(from_test_file, to_test_file, predictions_teacher_file, svocab, rev_svocab, tvocab, rev_tvocab, config, FLAGS.train_dir)

  elif FLAGS.decode:
    if FLAGS.from_test_data:
      from_test_file = FLAGS.from_test_data
      predictions_file = FLAGS.test_predictions + ".decoded"
    else:
      options_error("Arg error: test files must be given for decode and predict")
    print("Initializing vocabularies...")
    svocab, rev_svocab = reader_seq2seq._read_vocab(svocab_path)
    tvocab, rev_tvocab = reader_seq2seq._read_vocab(tvocab_path)
    print("Decoding test sentences...")
    config.batch_size = 1
    decode(from_test_file, predictions_file, svocab, rev_svocab, tvocab, rev_tvocab, config, FLAGS.train_dir)
    
  elif FLAGS.test:
    if FLAGS.from_test_data and FLAGS.to_test_data:
      from_test_file = FLAGS.from_test_data
      to_test_file = FLAGS.to_test_data
    else:
      options_error("Arg error: test files must be given for decode and predict")
    print("Initializing vocabularies...")
    svocab, rev_svocab = reader_seq2seq._read_vocab(svocab_path)
    tvocab, rev_tvocab = reader_seq2seq._read_vocab(tvocab_path)
    print("Reading test data...")
    test_dataset = reader_seq2seq._read_textdata(from_test_file, to_test_file, svocab, tvocab, config.encoder_size, config.decoder_size)
    print("Calculating perplexity...")
    config.batch_size = 1
    do_test(test_dataset, config, FLAGS.train_dir)

  else:
    if not (FLAGS.from_train_data and FLAGS.to_train_data):
      options_error("Arg error: training files not given")
      
    from_train_data = FLAGS.from_train_data
    to_train_data = FLAGS.to_train_data

    print("Creating vocabularies...")
    strain_words = reader_seq2seq._read_words(from_train_data)
    ttrain_words = reader_seq2seq._read_words(to_train_data)
    svocab, rev_svocab = reader_seq2seq._build_vocab(strain_words, FLAGS.from_vocab_size)
    tvocab, rev_tvocab = reader_seq2seq._build_vocab(ttrain_words, FLAGS.to_vocab_size)
    reader_seq2seq._write_vocab(svocab, svocab_path)
    reader_seq2seq._write_vocab(tvocab, tvocab_path)

    print("Reading training data...")
    train_dataset = reader_seq2seq._read_textdata(from_train_data, to_train_data, svocab, tvocab, config.encoder_size, config.decoder_size)
    if FLAGS.from_dev_data and FLAGS.to_dev_data:
      from_dev_data = FLAGS.from_dev_data
      to_dev_data = FLAGS.to_dev_data
      print("Reading dev data...")
      dev_dataset = reader_seq2seq._read_textdata(from_dev_data, to_dev_data, svocab, tvocab, config.encoder_size, config.decoder_size)
    else:
      dev_dataset = reader_seq2seq._read_textdata(from_train_data, to_train_data, svocab, tvocab, config.encoder_size, config.decoder_size)
    print("Training...")
    start_time = time.time()
    with tf.Graph().as_default():
      with tf.Session(config=get_gpu_config()) as session:
        s2s_model = create_model(session, False, True, config, FLAGS.train_dir)
        s2s_model.train(session, config, train_dataset, dev_dataset, exit_criteria, FLAGS.train_dir)
    print("Total time %s" % timedelta(seconds=time.time() - start_time))
    print("Done training!")


if __name__ == "__main__":
  tf.app.run()
