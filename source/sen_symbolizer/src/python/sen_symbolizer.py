#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Thu Jan  5 14:33:13 2017

Sequence2Sequence-based Symbolizer, exteded from tensorflow squence-to-sequence
model.
@author: Duc-Duy Nguyen

What is symbolization: the objective of symbolization share many similarity wi-
th lemmatization and normalization. The important advandates are
- Flexibility: geting rid of dictionary-based or rule-based approach, it emplo-
yed Neural Network on building model, which can be effectively improve perform-
ance after retraining as new samples came.
- Overcome error/mispelling: as an statistical approach, to a certain extend, 
it deals with errors well.
- Specified for Parallel Meaning Bank, but can be modified to work as a compon-
ent of any other system.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys,re,codecs
import logging
from tensorflow.python.platform import gfile

import numpy as np
import tensorflow as tf

#import data_utils
import seq2seq_model

_DIGIT_RE = re.compile(br"\d")

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

re.UNICODE = True;

tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.") #1024
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")

tf.app.flags.DEFINE_float("learning_rate", 0.3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")

tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_string("sem_file", "en.semtag", "Input semantic tags file") # Included tokens
tf.app.flags.DEFINE_integer("wrd_vocab_size", 1000, "Input vocabulary size.")
tf.app.flags.DEFINE_integer("sym_vocab_size", 1000, "Output vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "tmp", "Training directory.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(13,27),(36,35),(75,8),(85,70)]

###############################################################################

def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.wrd_vocab_size,
      FLAGS.sym_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    logging.info("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

###############################################################################

def addspaces(inpstr,spc):
    """Add space character after each character of input string"""
    if spc==0: # do nothing
        return inpstr;
    else:
        rs="";
        for c in inpstr:
            rs+= c + " ";
        return rs.strip();

###############################################################################

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

###############################################################################

def token_to_tokwrd_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string/character to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
    
  For character level, with vocabulary {"a": 1, "b": 2, "t":3} the tokenized l-
  ist ["b","a","t"] input will return [2,1,3]
  
  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    #print("!!!!!!!!!!!!{0}!!!!!!!!!!!!!!".format(sentence))
    words = basic_tokenizer(sentence)
    #print(words);
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  return [vocabulary.get(w, UNK_ID) for w in words]

###############################################################################

def basic_tokenizer(token):
  """Very basic tokenizer: split the word into a list of chars."""
  rs = [w for w in token.strip().split() if w];
  return rs

###############################################################################

def decode():
  """Perform the symbolization."""
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    wrd_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.word" % FLAGS.wrd_vocab_size)
    sym_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.sym" % FLAGS.sym_vocab_size)
    wrd_vocab, _ = initialize_vocabulary(wrd_vocab_path)
    _, rev_sym_vocab = initialize_vocabulary(sym_vocab_path)

    if not (gfile.Exists(FLAGS.sem_file)):
        logging.warning("Semantic file does not exist!")
        sys.exit()
    
    tokens = [];
    
    with codecs.open(FLAGS.sem_file, mode="rb" ,encoding='utf-8') as semtag_file:
        for line in semtag_file:
            line = line.strip("\n").encode('utf-8')
            line = re.sub(r'[“”«»]{3}', "\"", line)
            line = re.sub(r'[‘’‹›]{3}', "\'", line)
            #print(line)
            if line:
                semtag,text = line.split("\t");
                tokens.append("#"+semtag+ " ~ " + addspaces(text.lower(),1))
    # Decode from standard input.
    logging.info(tokens)
    symbols = []
    #tokens = ["#CON ~ m e n","#CLO ~ h a l f ~ p a s t ~ t w o"]; Test
    for token in tokens:
        # Get token-ids for the input token.
        logging.info("Processing token: " + token)
        tokwrd_ids = token_to_tokwrd_ids(tf.compat.as_bytes(token), wrd_vocab,tokenizer=None)
        if len(tokwrd_ids)<2:
            logging.warning("Invalid testing sample: {0}".format(tokwrd_ids))
        bucket_id = len(_buckets) - 1
        for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(tokwrd_ids):
                bucket_id = i
                break
            else:
                logging.warning("token truncated: %s", token) 

        # Get a 1-element batch to feed the token to the model.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              {bucket_id: [(tokwrd_ids, [])]}, bucket_id)
        # Get output logits for the token.
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if EOS_ID in outputs:
            outputs = outputs[:outputs.index(EOS_ID)]
        # Print out Output token corresponding to outputs.
        try:
            sym = "".join([tf.compat.as_str(rev_sym_vocab[output]) for output in outputs])
        except:
          sym = token.strip(" ")
        symbols.append(sym)
        sys.stdout.write(sym+"\n")
    logging.info(symbols)
      
###############################################################################
      
def main(_):
  decode()
  
if __name__ == "__main__":
  tf.app.run()
