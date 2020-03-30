# --------------------------------------
# Author: Annie Louis
#
# Reader for the seq2seq model. Generates batches as well. 
# ---------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import re
import numpy as np
import gzip
import tarfile
import getopt
import collections
from six.moves import urllib
from tensorflow.python.platform import gfile
import tensorflow as tf

UNKNOWN_WORD = "-UNK-"
EMPTY_WORD = "-EMP-"
END_SENT = "-eos-"
GO = "-GO-"

special_symbols = {EMPTY_WORD:0, UNKNOWN_WORD:1, END_SENT:2, GO:3}

def _read_words(filename):
  with tf.device('/cpu:0'):
    with tf.gfile.GFile(filename, "r") as f:
      return f.read().decode("utf-8").strip().split()


def _read_lines(filename):
  with tf.device('/cpu:0'):
    with tf.gfile.GFile(filename, "r") as f:
      ret = []
      for l in f:
        ret.append(l.decode("utf8").strip())
      return ret


def _build_vocab(list_words, vlimit):
  counter = collections.Counter(list_words)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  words = [w for (w, v) in count_pairs if w != END_SENT][0: (vlimit-len(special_symbols))]
  word_ids = [x + len(special_symbols) for x in range(len(words))]
  word_to_id = dict(zip(words, word_ids))
  id_to_word = dict(zip(word_ids, words))
  for (s, v) in special_symbols.iteritems():
    word_to_id[s] = v
    id_to_word[v] = s
  return word_to_id, id_to_word


def _read_vocab(filename):
  with tf.device('/cpu:0'):
    word_to_id = {}
    id_to_word = {}
    with tf.gfile.GFile(filename, "r") as ff:
      for line in ff:
        word, iden = line.strip().split('\t')
        word_to_id[word] = int(iden)
        id_to_word[int(iden)] = word
    return word_to_id, id_to_word

  
def _write_vocab(vocab, filename):
  with tf.device('/cpu:0'):
    with tf.gfile.GFile(filename, "w") as ff:
      for w, wid in vocab.iteritems():
        ff.write(w + "\t" + str(wid) + "\n")

        
def _get_ids_for_wordlist(list_words, word_to_id):
  ret = []
  for k in list_words:
    if k in word_to_id:
      ret.append(word_to_id[k])
    else:
      ret.append(word_to_id[UNKNOWN_WORD])
  return ret


def _get_id(word, word_to_id):
  if word in word_to_id:
    return word_to_id[word]
  return word_to_id[UNKNOWN_WORD]


def _get_empty_id():
  return special_symbols[EMPTY_WORD]


def _get_unknown_id():
  return special_symbols[UNKNOWN_WORD]


def _get_go_id():
  return special_symbols[GO]


def _read_textdata(from_file, to_file, from_vocab, to_vocab, encoder_size, decoder_size):
  dset = seq_dataset(encoder_size, decoder_size)
  with tf.device('/cpu:0'):
    with tf.gfile.GFile(from_file, mode="r") as source_file:
      with tf.gfile.GFile(to_file, mode="r") as target_file:
        source, target = source_file.readline(), target_file.readline()
        counter = 0
        while source and target:
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          source_ids = _get_ids_for_wordlist(source.split(), from_vocab)
          target_ids = _get_ids_for_wordlist(target.split(), to_vocab)
          dset.add_data(source_ids, target_ids)
          source, target = source_file.readline(), target_file.readline()
  return dset


def _dataset_from_lines(from_to_lines, from_vocab, to_vocab, encoder_size, decoder_size):
  dset = seq_dataset(encoder_size, decoder_size)
  for (source, target) in from_to_lines:
    source_ids = _get_ids_for_wordlist(source.split(), from_vocab)
    target_ids = _get_ids_for_wordlist(target.split(), to_vocab)
    dset.add_data(source_ids, target_ids)
  return dset


class seq_dataset(object):

  def __init__(self, ensize, desize):
    self.data = []
    self.encoder_size = ensize
    self.decoder_size = desize
    

  def add_data(self, sexample, texample):
    self.data.append((sexample, texample))

    
  def to_string(self, svocab_rev, tvocab_rev):
    rep = []
    for (source, target) in self.data:
      rep.append(str(([svocab_rev[s] for s in source], [tvocab_rev[t] for t in target])))
    return ' '.join(rep).strip()

  
  def batch_producer(self, bsize):
    """ Divides data into batches of size k. We also return 
    - how many dummy batch entries we had to create 
    - the ids into the data list, so that we can retrieve the source and target 
    sentences if needed (for results and display mostly during test time)
    """
    with tf.device('/cpu:0'):
      # Get a batch of encoder and decoder inputs from data,
      # pad them if needed, reverse encoder inputs and add GO to decoder.
      num_batches = len(self.data)// bsize
      left_over = len(self.data) - num_batches * bsize
      extra_one_batch = 0
      empty_sequences = 0
      if left_over > 0:
        extra_one_batch = 1
        empty_sequences = bsize - left_over
      for i in xrange(num_batches + extra_one_batch):
        encoder_inputs, decoder_inputs = [], []
        index_into_data = []
        for j in xrange(bsize):
          encoder_input, decoder_input = [], []
          if i * bsize + j < len(self.data):
            encoder_input, decoder_input = self.data[i * bsize + j]
            index_into_data.append(i * bsize + j)
          encoder_inputs.append(self.get_encoder_input_as_list(encoder_input, self.encoder_size)) # this is where encoder inputs are reversed
          decoder_inputs.append(self.get_decoder_input_as_list(decoder_input, self.decoder_size))
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = self.time_first_format(encoder_inputs, decoder_inputs, self.encoder_size, self.decoder_size, bsize)
        if left_over > 0 and i == num_batches: # the last batch  
          yield empty_sequences, index_into_data, batch_encoder_inputs, batch_decoder_inputs, batch_weights
        else:
          yield 0, index_into_data, batch_encoder_inputs, batch_decoder_inputs, batch_weights

          
  def get_encoder_input_as_list(self, encoder_input, encoder_size):
    """
    Encoder inputs are reversed and padded
    """
    encoder_pad = [_get_empty_id()] * (encoder_size - len(encoder_input)) 
    return list(reversed(encoder_input + encoder_pad))

  def get_decoder_input_as_list(self, decoder_input, decoder_size):
    """
    Decoder inputs get an extra "GO" symbol at the start, empty spaces are padded
    """
    decoder_pad_size = decoder_size - len(decoder_input) - 1  
    return [_get_go_id()] + decoder_input + [_get_empty_id()] * decoder_pad_size
  

  def time_first_format(self, encoder_inputs, decoder_inputs, encoder_size, decoder_size, batch_size):
    """
    Now we create matrices from the data selected above. The exact format is a list of length
    timesteps. Each element of this list is an array that is batch_size long
    """
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

      batch_weight = np.ones(batch_size, dtype=np.float32)       
      for batch_idx in xrange(batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == _get_empty_id():
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights


  
def main(argv):
  """ primarily to test if the readers work"""
  if len(argv) < 6:
      print("reader_seq2seq.py -f <from_file> -t <to_file> -v <vocab_limit_count> -b <batch_size> -e <encoder_size> -d <decoder_size>")
      sys.exit(2)

  from_file, to_file = "", ""
  batch_size  = 0
  vlimit = 0
  encoder_size, decoder_size = 0, 0 
  
  try:
    opts, args = getopt.getopt(argv, "f:t:v:b:e:d:")
  except getopt.GetoptError:
      print("reader_seq2seq.py -f <from_file> -t <to_file> -v <vocab_limit_count> -b <batch_size> -e <encoder_size> -d <decoder_size>")
      sys.exit(2)
  for opt, arg in opts:
    if opt == "-f":
      from_file = arg
    if opt == "-t":
      to_file = arg
    if opt == "-v":
      vlimit = int(arg)
    if opt == "-b":
      batch_size = int(arg)
    if opt == "-e":
      encoder_size = int(arg)
    if opt == "-d":
      decoder_size = int(arg)

  from_conts = _read_words(from_file)
  to_conts = _read_words(to_file)
  from_vocab, rev_from_vocab = _build_vocab(from_conts, vlimit)
  to_vocab, rev_to_vocab = _build_vocab(to_conts, vlimit)
  ip_dataset = _read_textdata(from_file, to_file, from_vocab, to_vocab, encoder_size, decoder_size)
  print("--From vocab--")
  print(str(from_vocab))
  print("--To vocab--")
  print(str(to_vocab))
  print("-- Input dataset --")
  print(ip_dataset.to_string(rev_from_vocab, rev_to_vocab))

  print("\n\nBatches from data ")
  for (empty, id_into_data, en_inputs, de_inputs, b_weights) in ip_dataset.batch_producer(batch_size):
    print("-- Batch --")
    print("empty seq = " + str(empty))
    print("ids into data = " + str(id_into_data))
    # en_str = []
    # for inp in en_inputs:
    #   en_str.append(str([rev_from_vocab[elem] for elem in inp]))
    # print("encoder:" + str(en_str))
    # de_str = []
    # for inp in de_inputs:
    #   de_str.append(str([rev_to_vocab[elem] for elem in inp]))
    # print("decoder:" + str(de_str))
    # print("weights:" + str(b_weights))

    # print("\nbatchwise data")
    b_weights_display = np.stack(b_weights, axis=1)
    en_inputs_display = np.stack(en_inputs, axis=1)
    de_inputs_display = np.stack(de_inputs, axis=1)
    en_str_b = []
    for i in range(len(en_inputs_display)):
      seq_rev = [rev_from_vocab[elem] for elem in en_inputs_display[i]]
      en_str_b.append(str(seq_rev[::-1]))
    print("encoder:" + str(en_str_b))
    de_str_b = []
    for i in range(len(de_inputs_display)):
      de_str_b.append(str([rev_to_vocab[elem] for elem in de_inputs_display[i]]))
    print("decoder:" + str(de_str_b))
    print("weights:" + str(b_weights_display))
    print("---------end of batch-----------\n")
    
if __name__=="__main__":
  main(sys.argv[1:])

