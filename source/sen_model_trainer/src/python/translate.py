# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#   Edited by Nguyen Duc-Duy (RUG) as PMB project. Jan 2017.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates word sentences into symbol.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from unidecode import unidecode

import os, sys, math, random, logging, time, timeit, re
ovr_start_time = timeit.default_timer()

from tensorflow.python.platform import gfile
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from threading import Thread
from collections import Counter;

import data_utils
import seq2seq_model

EXECUTION_TIME_LIMIT = 86400 # Time limit for program, in seconds. Default is 24 hours

tf.app.flags.DEFINE_float("learning_rate", 0.3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.") #1024
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("wrd_vocab_size", 1000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("sym_vocab_size", 1000, "French vocabulary size.")

tf.app.flags.DEFINE_string("data_dir", "tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "tmp", "Training directory.")
tf.app.flags.DEFINE_string("train_files_prefix", "data.train", "Prefix of training files name.")
tf.app.flags.DEFINE_string("dev_files_prefix", "data.train", "Prefix of dev files name.")
tf.app.flags.DEFINE_string("test_files_prefix", "data.test", "Prefix of test files name.")

tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 400,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("train", False,
                            "Set to True for interactive training.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("evaluate", False,
                            "Set to True for evaluating model with test set.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_integer("max_train_steps", 3200,
                            "Limit on the number of steps on training")



FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(13,27),(36,35),(75,8),(85,70)]
_buckets = [(37, 36), (45, 13), (73, 10), (85, 65)] 


###############################################################################

def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
          source = unidecode(source); # Convert source to nearest ASCII symbol
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            #if bucket_id == 4:
            #  print(str(source_ids) + " " + str(target_ids))
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

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
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

###############################################################################

def train():
  """Train a word->symbol translation model using WMT data."""
  # Prepare WMT data.
  print("Preparing data in %s" % FLAGS.data_dir)
  wrd_train, sym_train, wrd_dev, sym_dev, _, _ = data_utils.prepare_wmt_data(
      FLAGS.data_dir, FLAGS.wrd_vocab_size, FLAGS.sym_vocab_size,\
      FLAGS.train_files_prefix,FLAGS.dev_files_prefix)
  print ("wrd_train: {0}\nsym_train: {1}\nwrd_dev: {2}\nsym_dev: {3}\n".format(wrd_train,sym_train,wrd_dev,sym_dev));
  ### start session
  #config=tf.ConfigProto()
  # config.gpu_options.per_process_gpu_memory_fraction=0.98
 # config.gpu_options.allocator_type="BFC"
  #config.log_device_placement=True
 # with tf.Session(config=config) as sess:
  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(wrd_dev, sym_dev)
    train_set = read_data(wrd_train, sym_train, FLAGS.max_train_data_size)
    #print("train_set{0} \ndev_set: {1}".format(train_set,dev_set))
    print("Reading completed. Now estimating buckets and sizes.")
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))] # Number of document in every bucket type
    train_total_size= float(sum(train_bucket_sizes)) # Sum of all doc in any bucket size
    print("_buckets{0} \ntrain_bucket_sizes: {1} \ntrain_total_size: {2}".format(_buckets,train_bucket_sizes,train_total_size))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.

    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    
    STOP_SIGNAL = False;
    while not STOP_SIGNAL:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1
      max_preplexity_count = 0 # count number of time pre all reach 1.0
      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Record perplexity
        prexs = [0.0] * len(_buckets)
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            prexs[bucket_id]=1.0;
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          prexs[bucket_id]=eval_ppx;
        if max(prexs)<1.1:
          max_preplexity_count+=1;
        else:
          max_preplexity_count=0;
        print("max_preplexity_count: %d \n prexs: %s" % (max_preplexity_count,prexs))
        elapsed_time = timeit.default_timer() - ovr_start_time;
        if (current_step > max(train_total_size/FLAGS.batch_size,FLAGS.max_train_steps) 
            or max_preplexity_count==3 or elapsed_time>=EXECUTION_TIME_LIMIT):
           print("Finalizing the training session. Elapsed time: {0}.".format(elapsed_time))     
           STOP_SIGNAL = True;
           sess.close()
          
        sys.stdout.flush()

  print("Training Completed.")  
###############################################################################

def addspaces(inpstr,spc):
    if spc==0: # do nothing
        return inpstr;
    else:
        rs="";
        if re.match( r'^#[A-Z]{3}~', inpstr):
            rs = inpstr[:4]+" ~ ";
            inpstr = inpstr[5:]
        for c in inpstr:
            rs+= c + " ";
        return rs.strip();


###############################################################################

# Evaluating thread
def evaluator(threadID,inpData,model,result,sem_tags,errors,sess,wrd_vocab, rev_sym_vocab): # list of tuple (word,groundtruth)
    #print("Evaluator {0} received data size {1}".format(threadID,len(inpData)))
    if inpData:
      total_count=0;
      correct_count=0;
      error_semtag_counter = Counter(); # THis is to count err by truth semtag
      total_semtag_counter = Counter(); # this is to record occurences of semtages
      sem_tgs_rs = set([])
      for word,truth in inpData:
        total_count += 1;
        tokwrd_ids = word.split(' ');
        if len(tokwrd_ids)<2:
            print("Invalid testing sample: {0} -- {1}".format(word,truth))            
            continue;
        for i in xrange(len(tokwrd_ids)):
          tokwrd_ids[i] = tokwrd_ids[i].strip()
        #print(str(tokwrd_ids) + " -- >" + str(len(tokwrd_ids)));
        # Which bucket does it belong to?                                                    
        bucket_id = len(_buckets) - 1                                                        
        for i, bucket in enumerate(_buckets):
          #print(str(i) + " " + str(bucket))
          if bucket[0] >= len(tokwrd_ids):                                                    
            bucket_id = i                                                                   
            break                                                                            
        else:                                                                                
          logging.warning(" Thread " + str(threadID) + " says: Sentence truncated: %s", word)                                    
        # Get a 1-element batch to feed the sentence to the model.                           
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(                    
                  {bucket_id: [(tokwrd_ids, [])]}, bucket_id)                                       
        # Get output logits for the sentence.                                                
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,               
                                                 target_weights, bucket_id, True)                    
        # This is a greedy decoder - outputs are just argmaxes of output_logits.             
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]                 
        # If there is an EOS symbol in outputs, cut them at that point.                      
        if data_utils.EOS_ID in outputs:                                                     
          outputs = outputs[:outputs.index(data_utils.EOS_ID)]                               
        # Print out French sentence corresponding to outputs.                                
        rs = " ".join([tf.compat.as_str(rev_sym_vocab[output]) for output in outputs])        
        #print("%s -- %s" % rs,num); 
        #rs_semtag = word[:2];
        rs_semtag = tokwrd_ids[0];
        sem_tgs_rs.add(rs_semtag);
        total_semtag_counter[rs_semtag]+=1;                                   
        if rs == truth:                                                                        
          correct_count+=1.0;
        else: # Record error
          error_semtag_counter[rs_semtag]+=1;
            
        if total_count%1000 ==0:
          print("Evaluator "+ str(threadID)+ " processed " + str(total_count) + " recs.")
    print('Evaluator: ' + str(threadID) + ' finished. Result: '+  str( correct_count) + " correct out of " + str(len(inpData)))
    # Report results
    result[threadID] = (correct_count,len(inpData))
    errors[threadID] = (error_semtag_counter,total_semtag_counter)
    sem_tags[threadID]= sem_tgs_rs;

###############################################################################
    
def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    wrd_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.word" % FLAGS.wrd_vocab_size)
    sym_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.sym" % FLAGS.sym_vocab_size)
    wrd_vocab, _ = data_utils.initialize_vocabulary(wrd_vocab_path)
    _, rev_sym_vocab = data_utils.initialize_vocabulary(sym_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = addspaces(sys.stdin.readline(),1); #sentence = sys.stdin.readline(); 
    print(sentence)
    while sentence:
      # Get token-ids for the input sentence.
      tokwrd_ids = data_utils.sentence_to_tokwrd_ids(tf.compat.as_bytes(sentence), wrd_vocab)
      # Which bucket does it belong to?
      bucket_id = len(_buckets) - 1
      for i, bucket in enumerate(_buckets):
        if bucket[0] >= len(tokwrd_ids):
          bucket_id = i
          break
        else:
          logging.warning("Sentence truncated: %s", sentence) 

      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(tokwrd_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_sym_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = addspaces(sys.stdin.readline(),1)

###############################################################################

def evaluate():
    # run evaluation
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    wrd_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.word" % FLAGS.wrd_vocab_size)
    sym_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.sym" % FLAGS.sym_vocab_size)
    wrd_vocab, _ = data_utils.initialize_vocabulary(wrd_vocab_path)
    _, rev_sym_vocab = data_utils.initialize_vocabulary(sym_vocab_path)


    #sentence = sys.stdin.readline(); 
    # Get address
    test_name = FLAGS.test_files_prefix;
    #test_name = "pmb_2016-12-14_sembows_only_filtered"
    test_path = os.path.join(FLAGS.data_dir, test_name)
    print("~~~~~~~~~~~~~" , test_path,)
    # Create token ids for the test data.
    target_test_ids_path = test_path + (".ids%d.sym" % FLAGS.sym_vocab_size)
    source_test_ids_path = test_path + (".ids%d.word" % FLAGS.wrd_vocab_size)
    data_utils.data_to_tokwrd_ids(test_path + ".sym", target_test_ids_path, sym_vocab_path, tokenizer=None)
    data_utils.data_to_tokwrd_ids(test_path + ".word", source_test_ids_path, wrd_vocab_path, tokenizer=None)    
    
    num_path = test_path + ".sym";
    word_path = test_path + ".word";
    if not (gfile.Exists(num_path) and gfile.Exists(word_path)):
        print("Test file(s) are not found. Check the data folder.")
        sys.exit()
    print("Started evaluating model on test data.")
    # Build the test set
    test_set = read_data(source_test_ids_path, target_test_ids_path);

    """
    for bucket_id in xrange(len(_buckets)):
      if len(test_set[bucket_id]) == 0:
        print("  eval: empty bucket %d" % (bucket_id))
        continue
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
         test_set, bucket_id)
      _, eval_loss, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
      print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
    sys.stdout.flush()
    """
    
    NUM_EVALUATOR = 40 # number of evaluation threads
    threads = [None] * NUM_EVALUATOR;
    results = [None] * NUM_EVALUATOR; # format: (corect_count,total_count)
    sem_tags = [None] * NUM_EVALUATOR;
    errors = [None] * NUM_EVALUATOR;  # format: (error_COUNTER, total_COUNTER)  by semantic tages
    test_set = [None]* NUM_EVALUATOR; # List to save input data for each thread, which is a list of pairs (sentence,truth value)

    # Read these two test file
    test_size=0;
    with gfile.GFile(source_test_ids_path, mode="rb") as word_f_ids:
         word_count = -1;
         num_count = -1;
         with gfile.GFile(num_path, mode="rb") as num_f:
             irofile = iter(num_f);
             for word in word_f_ids: 
                word_count += 1;
                test_size = word_count;
                num_count +=1;
                num = next(irofile).strip();
                word = word.strip();
                index = word_count%NUM_EVALUATOR;
                #print(index)
                if test_set[index] == None:
                  test_set[index] = [(word,num)]
                else:
                  test_set[index].append((word,num))
                #print(test_set);
                #if word_count ==29:
                #  break;
             # Close files
             num_f.close();
             word_f_ids.close();
    # Now call the evaluators
    for i in range(len(threads)):
      threads[i] = Thread(target=evaluator, args=(i,test_set[i],model,results,sem_tags,errors,sess,wrd_vocab, rev_sym_vocab))
      threads[i].start()
      print("Started thread " + str(i) + " with " + str(len(test_set[i])) + " records.")
    
    # Prepare tag dictionary while the threads are running
    #sym_vocab_path = "vocab1000.sym"
    word_dic = None; # Save the symbol dictionary
    with gfile.GFile(wrd_vocab_path, mode="rb") as sym_vocab_file:
      lns = [line.strip() for line in sym_vocab_file.readlines()]  
      word_dic = enumerate(lns)
    
    for i in range(len(threads)):
      threads[i].join()
    
    # Collect and report RESULT matrix
    results = np.array(results)
    print("RESULT REPORT:\N - Result matrix: \n" + str(results))
    print(" - Sum: " + str(np.sum(results,axis=0)))
    total_correct, total_count = np.sum(results,axis=0)[0],np.sum(results,axis=0)[1]
    #print(results)
    """total_correct = 0;
    total_count =0;
    for correct,total in results:
      total_correct+= correct;
      total_count+=total;total_correct"""
    #Report
    print(" - Correct: {0} out of {1}".format(total_correct,total_count))
    print(" - Accuracy: {0}".format( str(total_correct/total_count)))
    # Verify result size and input size
    if (total_count!=(test_size+1)):
      print("[E] Size of test file ({0}) and reported result ({1}) is MISMATCHED!".format(total_count,test_size))
     
    # Collect and report ERROR matrix
    # First get list of semtag recorded by thread o
    tags_ls = set([item for sublist in sem_tags for item in sublist]); # get the total tages
    if tags_ls:
        tags_ls = list(tags_ls); # convert to list to keep order
        # Now we build a matrix to store error report, where rows is threadID
        # (0-NUM_EVALUATOR) and columns are semantic tages (in tags_ls). Cell
        # store the correct rate 
        error_list = [];
        sum_error_cnt = errors[0][0];
        sum_total_cnt = errors[0][1];
        i = 0;
        for  error_semtag_counter,total_semtag_counter in errors:
            print("ThdID: {0}, ErrCnt: {1}".format(i,error_semtag_counter));
            print("ThdID: {0}, TotCnt: {1}".format(i,total_semtag_counter))
            row = [];            
            for tag in tags_ls:
                if total_semtag_counter[tag]!=0:
                    row.append(error_semtag_counter[tag]/total_semtag_counter[tag]);
                else:
                    row.append(-1);
            row = tuple(row);
            error_list.append(row);
            if i > 0:
                sum_error_cnt = sum_error_cnt + error_semtag_counter;
                sum_total_cnt = sum_total_cnt + total_semtag_counter;
            print("ThdID: {0}, sumErrCnt: {1}".format(i,sum_error_cnt));
            print("ThdID: {0}, sumTotCnt: {1}".format(i,sum_total_cnt))
            
            i+=1;
        error_matrix = np.array(error_list);
        print(" - Calculated error matrix size: {0}.".format(str(error_matrix.shape)))
        np.set_printoptions(threshold=np.inf);
        print(" - Error matrix: {0}".format(str(error_list)));
        # now print the report for each tages:
        print(" - Error break down:");        
        word_dic = dict(word_dic)
        print(word_dic)
        for tag in tags_ls: 
            val = sum_error_cnt[tag]/sum_total_cnt[tag] if sum_total_cnt[tag]!=0 else "NA";
            e_val = sum_error_cnt[tag];
            t_val = sum_total_cnt[tag];
            print("   + Tag: {0} has error rate: {1} ({2} out of {3})".format(word_dic[int(tag)],val,e_val,t_val))
        
    else: # some tags in err tages doesn't appear in total tag
        print("[E] Empty tag list! Somthing is wrong!")

###############################################################################

def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)

###############################################################################
###############################################################################

def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  elif FLAGS.train and FLAGS.evaluate:
    train()
    print("<!> Train completed. Starting evaluation.")
    evaluate()
  elif FLAGS.evaluate:
    evaluate()
    #print("!!!!!!!!!!!")
  else:
    train()
    
###############################################################################
if __name__ == "__main__":
  tf.app.run()
