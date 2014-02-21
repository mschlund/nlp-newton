import nltk
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from nltk import treetransforms
import pickle
import time
import subprocess
import shlex
import re
import os
import sys

#treebanks are expected to reside e.g. in the ~/nltk_data/corpora/ directory
#or in any other directory that is in nltk.data.path --- note that nltk always expects a subdirectory named "corpora" which is a bit unnerving!!
#please adapt all paths to your system as needed!
prefix = "/home/" #TODO: adapt this to your system!
parser_path = prefix + "stanford-parser/stanford-parser.jar"
this_path = prefix + "nlp-newton"

#TODO: nltk needs all treebanks in "~/nltk_data/corpora/" (resp. adapted path)
#sample from the penn-treebank that is shipped with nltk (~3900 trees only)
pennbank = nltk.corpus.LazyCorpusLoader('treebank/combined', nltk.corpus.BracketParseCorpusReader, r'wsj_.*\.mrg', tag_mapping_function=nltk.corpus.simplify_wsj_tag)

# Version 8.0 of the Tuebingen TueBa/DZ treebank (75408 trees)
tubank = nltk.corpus.LazyCorpusLoader('tuebadz', nltk.corpus.BracketParseCorpusReader, r'tuebadz-utf8-r8.0.penn', comment_char='-',detect_blocks='sexpr')

#other treebanks to play around with in the future:
#tigerbank = nltk.corpus.LazyCorpusLoader('tiger', nltk.corpus.BracketParseCorpusReader, r'tiger_.*\.penn', comment_char='%')
#icebank = nltk.corpus.LazyCorpusLoader('icepahc-v0.9/psd', nltk.corpus.BracketParseCorpusReader, r'2008.*\.psd') #lacks S symbols...
#german_spmrl = nltk.corpus.LazyCorpusLoader('SPMRL/GERMAN_SPMRL/gold/ptb/train', nltk.corpus.BracketParseCorpusReader, r'train.German.gold.ptb')
#french_spmrl = nltk.corpus.LazyCorpusLoader('SPMRL/FRENCH_SPMRL/gold/ptb/train', nltk.corpus.BracketParseCorpusReader, r'train.French.gold.ptb')
#hungarian_spmrl = nltk.corpus.LazyCorpusLoader('SPMRL/HUNGARIAN_SPMRL/gold/ptb/train', nltk.corpus.BracketParseCorpusReader, r'train.Hungarian.gold.ptb')
#korean_spmrl = nltk.corpus.LazyCorpusLoader('SPMRL/KOREAN_SPMRL/gold/ptb/train', nltk.corpus.BracketParseCorpusReader, r'train.Korean.gold.ptb')
#basque_spmrl = nltk.corpus.LazyCorpusLoader('SPMRL/BASQUE_SPMRL/gold/ptb/train', nltk.corpus.BracketParseCorpusReader, r'train.Basque.gold.ptb')
#polish_spmrl = nltk.corpus.LazyCorpusLoader('SPMRL/POLISH_SPMRL/gold/ptb/train', nltk.corpus.BracketParseCorpusReader, r'train.Polish.gold.ptb')
#swedish_spmrl = nltk.corpus.LazyCorpusLoader('SPMRL/SWEDISH_SPMRL/gold/ptb/train5k', nltk.corpus.BracketParseCorpusReader, r'train5k.Swedish.gold.ptb')
#hebrew_spmrl = nltk.corpus.LazyCorpusLoader('SPMRL/HEBREW_SPMRL/gold/ptb/train5k', nltk.corpus.BracketParseCorpusReader, r'train5k.Hebrew.gold.ptb')


#compute the "dimension" (aka Horton-Strahler number) of a derivation tree as follows:
# a leaf has dimension 0
# if there are at least two children with maximal dimension the dimension increases by one
# otherwise the dimension is the maximum of the children's dimension
def dim(tree) :
  try :
    if not isinstance(tree, nltk.tree.Tree) :
      return 0
    else :
      cdims = sorted((dim(c) for c in tree), reverse=True)
      if len(cdims)==1 :
        return cdims[0]
      if cdims[0] == cdims[1] :
        return cdims[0] + 1
      else :
        return cdims[0]
  except :
    pass


def cut_tag (tag) :
  new = tag
  if '#' in new and not new.startswith('#'):
    new = new.split('#')[0]
  if '^' in new :
    new = new.split('^')[0]
  if '-' in new and not new.startswith('-'):
    new = new.split('-')[0]
  if '+' in new :
    new = new.split('+')[0]
  if ':' in new and not new.startswith(':'):
    new = new.split(':')[0]
  return new

def cut_tag_old (tag) :
  new = tag
  if '^' in new :
    new = new.split('^')[0]
  if '-' in new :
    new = new.split('-')[0]
  if '+' in new :
    new = new.split('+')[0]
  return new


def cut_tags(tree) :
  if not isinstance(tree, nltk.tree.Tree) :
    return tree #do not cut leaves!
  else :
    return nltk.Tree(cut_tag(tree.node), [cut_tags(c) for c in tree]) 

def cut_tags_old(tree) :
  if not isinstance(tree, nltk.tree.Tree) :
    return cut_tag_old(tree)
  else :
    return nltk.Tree(cut_tag_old(tree.node), [cut_tags_old(c) for c in tree]) 


# filter out NONE-tags
def filter_tree(tree) :
  if not isinstance(tree, nltk.tree.Tree):
    return tree
  else:
    if tree.node in ['-NONE-']:
      return None
    filtered_children = filter(lambda x: x!=None,[filter_tree(c) for c in tree])
    if filtered_children == [] :
      return None
    else :
      return nltk.Tree(tree.node, filtered_children)

# plot a joint histogram of the dimension and heights of all trees in the treebank supplied
# also calculate the average and maximum dimension/height and the empirical standard deviation
def get_dim_height_histo(tb) :
  dimlist = [dim(t) for t in tb.parsed_sents()]
  avg_dim = round(np.average(dimlist),2)
  stddev_dim = round(np.std(dimlist),2)
  max_dim = max(dimlist)

  list_height = [t.height() for t in tb.parsed_sents()]
  avg_height = round(np.average(list_height),2)
  stddev_height = round(np.std(list_height),2)
  max_height = max(list_height)

  f, axarr = plt.subplots(2, sharex=False)
  axarr[0].hist(dimlist, bins=[0,1,2,3,4,5,6])
  axarr[0].set_title('Dimension (avg,stdev,max): ' + str((avg_dim,stddev_dim,max_dim)))
  axarr[1].hist(list_height, bins=range(31))
  axarr[1].set_title('Height (avg,stdev,max): ' + str((avg_height,stddev_height,max_height)))
  plt.show()

# get all trees having dimension d --- useful for extracting all dimension-4 sentences for example
def get_dim_sents(tb,d):
  dimlist = [dim(t) for t in tb.parsed_sents()]
  hd = [tb.parsed_sents()[i] for i in [i for i,k in enumerate(dimlist) if k==d] ]
  hd_s = [' '.join(t.leaves()) for t in hd]
  return hd_s

def ignore_tag(t) :
  if t in ['-NONE-'] :
    return True
  else :
    return False

def ignore_punct_tag(t) :
  if t in ['-NONE-' , ',' , '.' , '!' , '?', '-','--', ':' , ';' , '``', '\'\''] or t.startswith('*') :
    return True
  else :
    return False

def filter_strict(tree) :
  if not isinstance(tree, nltk.tree.Tree):
    return tree
  else:
    if ignore_punct_tag(tree.node):
      return None
    filtered_children = filter(lambda x: x!=None,[filter_strict(c) for c in tree])
    if filtered_children == [] :
      return None
    else :
      return nltk.Tree(tree.node, filtered_children)


def cut_terminals(tree) :
  if len(tree) == 1 and not isinstance(tree[0], nltk.tree.Tree):
    return tree.node
  else :
    return nltk.Tree(tree.node, [cut_terminals(c) for c in tree])

# decorate all nodes with "-D" where D is the dimension of that node
# then parse the tag sample (with "-0" appended to each!)
def decorate_tree_dim(tree) :
  if not isinstance(tree, nltk.tree.Tree) :
    return tree
  else :
    cdims = sorted([dim(c) for c in tree], reverse=True)
    if len(cdims)==1 :
      tdim = cdims[0]
    elif cdims[0] == cdims[1] :
      tdim = cdims[0] + 1
    else :
      tdim = cdims[0]
    return nltk.Tree(tree.node+'+'+str(tdim), [decorate_tree_dim(c) for c in tree])

# decorate all nodes with "-H" where H is the height of that node
# note that nltk assigns height 1 to leaves normally (hence we use this strange workaround)
def decorate_tree_height(tree) :
  if not isinstance(tree, nltk.tree.Tree) :
    return tree
  else :
    cheights = sorted([c.height() for c in tree if isinstance(c,nltk.tree.Tree)], reverse=True)
    if [] == cheights :
      return nltk.Tree(tree.node+'+1', [decorate_tree_height(c) for c in tree])
    else :
      t_height = cheights[0] # t.height() actually gives us height+1 (if we define the height of a leaf as 0)
      return nltk.Tree(tree.node+'+'+str(t_height), [decorate_tree_height(c) for c in tree])


def parent_annot(tree, parent):
  if not isinstance(tree, nltk.tree.Tree) :
    return tree
  else:
    return nltk.Tree(tree.node+'+'+str(parent), [parent_annot(c, tree.node) for c in tree])

#parent annotations (not used right now since it is done by the Stanford-parser)
def v_markov(tree) :
  return nltk.Tree(tree.node, [parent_annot(tree[0], tree.node)])


# to be called for decorated trees only. Introduce a new root vertex named 'SDIMNEW'.
def make_new_root(tree) :
  return nltk.Tree('SDIMNEW', [tree])


# Our Strategy:
# (1) Write training-data to training-file
# (2) Write tagged test-sentences to test-file
# (3) Call stanford-parser (from this script), force it to use the tagged data and write results to outfile
# (4) Read in parsed-trees and evaluate them again using the Stanford parser's eval metrics (LP/LR/F1, Crossing Brackets, Leaf-Ancestor)
# (5) Try the same with annotated dimensions/heights/etc. (remember to annotate the test-tags with dim 0 !)

# remove all chains of unary roots from the top
# for dim-annotated-tree this amounts to: cut away ROOT-tag (generated by the stanford-parser) and additional root (SDIMNEW)
def normalize_parsed(tree) :
  t = tree
    # decend the tree and cut away unary roots until we hit the label 'VROOT' or 'TOP'
  while isinstance(t[0], nltk.tree.Tree) and len(t) == 1 and not t.node in ['VROOT', 'TOP'] :
    t = t[0]
  if t.node == 'TOP' : #rename the label to be compatible with the TueBa/DZ-TB
    t.node = 'VROOT'
  return t


# use the Stanford-parser's evalb method to assess P/R/F1
# we also have the option to use crossing-bracket-eval or leaf-ancestor eval as implemented in the Stanford-parser!
# write trees to files, then call evalb on them
def eval_parses(tb, seed, sample_size, unfold_dim = False, unfold_height = False, HMarkov = 0, VMarkov = 0, eval_method = "evalb") :
  print "Evaluating parses..."
  gold_filename = "gold_trees/gold_" + str(sample_size) + '_' + ('D_' if unfold_dim else '') + ('H_' if unfold_height else '') + str(seed)+('_VM'+str(VMarkov) if VMarkov>0 else '') + ('_HM'+str(HMarkov) if HMarkov>0 else '') + ".txt"
  parsed_filename = "guess_trees/guess_" + str(sample_size) + '_' + ('D_' if unfold_dim else '') + ('H_' if unfold_height else '') + str(seed)+('_VM'+str(VMarkov) if VMarkov>0 else '') + ('_HM'+str(HMarkov) if HMarkov>0 else '') + ".txt"
  
  if tb == "TueBaDZ":
    langswitch = " -l TueBaDZ "
  elif tb == "Tiger":
    langswitch = " -l German "
  elif tb == "SPMRL_FRENCH":
    langswitch = " -l French "
  else :
    langswitch = " "
  
  if eval_method == "evalb" :
    cmd_str = "java -cp " + parser_path + " -mx1500m edu.stanford.nlp.parser.metrics.Evalb " + langswitch +gold_filename + " " + parsed_filename
  elif eval_method == "CB" :
    cmd_str = "java -cp " + parser_path + " -mx1500m edu.stanford.nlp.parser.metrics.Evalb -b " + langswitch +gold_filename + " " + parsed_filename  
  elif eval_method == "LA" :
    #somehow the LA-metric only works with Negra-preprocessed trees (or similar)... if language switch is "TueBaDZ" this gives an exception since the tree-labels are not of Type HasIndex ..
    cmd_str = "java -cp " + parser_path + " -mx1500m edu.stanford.nlp.parser.metrics.LeafAncestorEval " + " -l German " +gold_filename + " " + parsed_filename
  
  args = shlex.split(cmd_str)
  p1 = subprocess.Popen(args,stdout=subprocess.PIPE)
  if eval_method == "evalb" :
    p2 = subprocess.Popen(shlex.split('grep summary'),stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1.stdout.close()
    
    out, err = p2.communicate()
    print str(out)
    # get the numbers (LP/LR/exact) from stdout of the process and print it to statfile
    nums = re.findall("\d+.\d+", str(out))
    nums = map(float,nums)
    prec,rcl,f1,ex = nums[0],nums[1],nums[2],nums[3] #nums[3] is the number of exactly parsed sentences
    stat_filename = tb + "_evalb_stats_"+('D_' if unfold_dim else '') +('H_' if unfold_height else '') + str(seed)+('_VM'+str(VMarkov) if VMarkov>0 else '') + ('_HM'+str(HMarkov) if HMarkov>0 else '') + ".txt"
    statfile = open("output_data/" + stat_filename,"a")
    statfile.write(str(sample_size) + " " + str(prec) + " " + str(rcl) + " " + str(f1) +  " " + str(ex) + '\n')
    statfile.close()
  elif eval_method == "CB" :
    p2 = subprocess.Popen(shlex.split('grep CBEval'),stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1.stdout.close()
    
    out, err = p2.communicate()
    print str(out)
    # get the numbers (average CB and num of sents with zero CB) from stdout of the process and print it to statfile
    nums = re.findall("\d+.\d+", str(out))
    nums = map(float,nums)
    avg,zero = nums[0],nums[1] #nums[0]= avg crossing brackets, nums[1] = zero CB
    stat_filename = tb + "_CBEval_stats_"+('D_' if unfold_dim else '') +('H_' if unfold_height else '') + str(seed)+('_VM'+str(VMarkov) if VMarkov>0 else '') + ('_HM'+str(HMarkov) if HMarkov>0 else '') + ".txt"
    statfile = open("output_data/" + stat_filename,"a")
    statfile.write(str(sample_size) + " " + str(avg) + " " + str(zero) + '\n')
    statfile.close()
  elif eval_method == "LA" :
    p2 = subprocess.Popen(shlex.split('grep Sentence-level -A 5'),stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1.stdout.close()
    
    out, err = p2.communicate()
    print str(out)
    # get the numbers (precisions) from stdout of the process and print it to statfile
    nums = re.findall("\d+.\d+", str(out))
    nums = map(float,nums)
    sent_lvl_avg, sent_lvl_ex, corp_lvl_avg = nums[0],nums[1],nums[2] #sentence level average+exact, corpus level average
    stat_filename = tb + "_LA_stats_"+('D_' if unfold_dim else '') +('H_' if unfold_height else '') + str(seed)+('_VM'+str(VMarkov) if VMarkov>0 else '') + ('_HM'+str(HMarkov) if HMarkov>0 else '') + ".txt"
    statfile = open("output_data/" + stat_filename,"a")
    statfile.write(str(sample_size) + " " + str(sent_lvl_avg) + " " + str(sent_lvl_ex) + " " + str(corp_lvl_avg) + '\n')
    statfile.close()
    
  return


# the parameters tb,seed,sample_size are there for naming input/output-files suitably
# training_sample holds a list of trees used for training, test_sample analogously
# tag_sample is a list of pairs (word, gold_tag) and hold the (gold-)tags that should be used by the parser
# the other parameters control the use of annotations
def parse_pcfg_treebank(tb, seed, sample_size, training_sample, test_sample, tag_sample, unfold_dim = False, unfold_height = False, HMarkov = 0, VMarkov = 0) :
  
  training_file_name = "training_data/training_" + str(sample_size) + '_' + ('D_' if unfold_dim else '') + ('H_' if unfold_height else '') + str(seed)+('_VM'+str(VMarkov) if VMarkov>0 else '') + ('_HM'+str(HMarkov) if HMarkov>0 else '') + ".txt"

  result_file_name = "parsed_" + str(sample_size) + '_' + ('D_' if unfold_dim else '') + ('H_' if unfold_height else '') + str(seed)+('_VM'+str(VMarkov) if VMarkov>0 else '') + ('_HM'+str(HMarkov) if HMarkov>0 else '') + ".txt"

  test_file_name = "test_sents/test-sents_" + str(sample_size) + '_' + ('D_' if unfold_dim else '') + ('H_' if unfold_height else '') + str(seed)+('_VM'+str(VMarkov) if VMarkov>0 else '') + ('_HM'+str(HMarkov) if HMarkov>0 else '') + ".txt"

  parsed_filename = "guess_trees/guess_" + str(sample_size) + '_' + ('D_' if unfold_dim else '') + ('H_' if unfold_height else '') + str(seed)+('_VM'+str(VMarkov) if VMarkov>0 else '') + ('_HM'+str(HMarkov) if HMarkov>0 else '') + ".txt"

  gold_filename = "gold_trees/gold_" + str(sample_size) + '_' + ('D_' if unfold_dim else '') + ('H_' if unfold_height else '') + str(seed)+('_VM'+str(VMarkov) if VMarkov>0 else '') + ('_HM'+str(HMarkov) if HMarkov>0 else '') + ".txt"

  speed_filename = "output_data/speeds_"+('D_' if unfold_dim else '') + ('H_' if unfold_height else '') + str(seed)+ ('_VM'+str(VMarkov) if VMarkov>0 else '') + ('_HM'+str(HMarkov) if HMarkov>0 else '') + ".txt"

  size_filename = "output_data/size_"+('D_' if unfold_dim else '') + ('H_' if unfold_height else '') + str(seed)+ ('_VM'+str(VMarkov) if VMarkov>0 else '') + ('_HM'+str(HMarkov) if HMarkov>0 else '') + ".txt"

  #if os.path.isfile(parsed_filename) :
  #  print "Parses were already done before... doing nothing"
  #  return

  training_file = open(training_file_name, 'wb')
  print('Creating training file...')
  for tree in training_sample:
    t_tmp = copy.deepcopy(tree)
    t_cut = cut_tags(t_tmp)
    
    # NOTE: we either unfold w.r.t. dim OR w.r.t. height !!
    if unfold_dim :
      t_unf = make_new_root(decorate_tree_dim(t_cut))
    elif unfold_height :
      t_unf = make_new_root(decorate_tree_height(t_cut))
    else :
      t_unf = t_cut

    t_training = t_unf
    training_file.write(t_training.pprint(margin=9999999)+ '\n')
  training_file.close()
  
  print('Creating test file...')
  test_file = open(test_file_name, 'wb')
  for s in tag_sample:
    if unfold_dim:
      s = [(x,st+'+0') for (x,st) in s]
    elif unfold_height:
      s = [(x,st+'+1') for (x,st) in s]
    sent = ' '.join([nltk.tag.tuple2str(w) for w in s])
    test_file.write(sent+'\n')
  
  test_file.close()
  
  #Determine tlpp according to the treebank! (Treebank Language Parser Parameters for the Stanford-parser)
  if tb == "TueBaDZ" :
    tlpp = "edu.stanford.nlp.parser.lexparser.TueBaDZParserParams"
  elif tb == "Tiger" :
    tlpp = "edu.stanford.nlp.parser.lexparser.NegraPennTreebankParserParams"
  elif tb == "SPMRL_FRENCH":
    tlpp = "edu.stanford.nlp.parser.lexparser.FrenchTreebankParserParams"
  elif tb == "Penn" :
    tlpp == "edu.stanford.nlp.parser.lexparser.EnglishTreebankParserParams"
  else : #default...
    tlpp = "edu.stanford.nlp.parser.lexparser.TueBaDZParserParams"
  
  VMarkov_str = " -vMarkov " + str(VMarkov+1) + " " if VMarkov >0 else " -vMarkov 1 " # somehow vMarkov 2 means parent annotation...
  HMarkov_str = " -hMarkov " + str(HMarkov) + " -compactGrammar 0 " if HMarkov >0 else " " #compactGrammar yields an exception otherwise---this switches it off!
  
  cmd_str = "java -cp " + parser_path + " -mx3000m edu.stanford.nlp.parser.lexparser.LexicalizedParser -tlPP " + tlpp + " -PCFG " + VMarkov_str + HMarkov_str + " -uwm 0 -headFinder edu.stanford.nlp.trees.LeftHeadFinder -sentences newline -smoothTagsThresh 0 -scTags -train " + training_file_name + " -tokenized -tagSeparator / -tokenizerFactory edu.stanford.nlp.process.WhitespaceTokenizer -tokenizerMethod newCoreLabelTokenizerFactory " + test_file_name
  print cmd_str

  args = shlex.split(cmd_str)
  resultfile = open("corpora/parsed_trees/" + result_file_name, 'wb')
  print('Parsing test-sentences...')
  p1 = subprocess.Popen(args,stdout=resultfile,stderr=subprocess.PIPE)
  
  out_p1, err_p1 = p1.communicate() # this is VERY important! we have to wait for the process p1 to finish!!
  # out_p1 contains the parsed trees, err_p1 contains debug/status/config messages from the parser (training parameters, size of the grammar, avg parsing time)
  print err_p1
  print out_p1

  record_grammar_size = True # hardcoded for now... ugly .. TODO: make parameter
  record_speeds = True

  if record_speeds:
    p2 = subprocess.Popen(shlex.split('grep "wds/"'),stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p2.communicate(err_p1)
    #print out
    nums = re.findall("\d+\.?\d*", str(out))
    nums = map(float,nums)
    print nums
    # get the result (wds/sec, sents/sec) from stdout of the process and return it
    speedfile = open(speed_filename,"a")
    speedfile.write(str(sample_size) + " " + str(nums[2]) + " " + str(nums[3]) + '\n') #the num of words is nums[0] and num of sents is in nums[1]
    p2.stdout.close()
    speedfile.close()
  if record_grammar_size:
    p2 = subprocess.Popen(shlex.split('grep "Grammar[[:space:]][[:digit:]]"'),stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p2.communicate(err_p1)
    #print out
    nums = re.findall("\d+", str(out))
    nums = map(int,nums)
    numstr = " ".join([str(x) for x in nums])
    size = nums[3]+nums[4] # = num UnaryRules + num BinaryRules
    print nums
    # get the result ("States	Tags	Words	UnaryR	BinaryR	Taggings") from stdout of the process and return it
    sizefile = open(size_filename,"a")
    sizefile.write(str(sample_size) + " " + str(size) + '\n')
    p2.stdout.close()
    sizefile.close()

  resultfile.close()
  nltk.data.path.append(this_path) # tell nltk to search this directory for the output/parsed-corpus
  cr = nltk.corpus.LazyCorpusLoader('parsed_trees', nltk.corpus.BracketParseCorpusReader, result_file_name)
  print cr
  
  print "Writing gold-trees to file..."
  goldfile = open(gold_filename,'wb')
  for t in test_sample :
    goldfile.write(normalize_parsed(cut_tags(t)).pprint(margin=9999999)+'\n') #margin is just ridiculously high so that one tree is written per line (and not indented)
  goldfile.close()

  print "Writing guess-trees to file..."
  parsedfile = open(parsed_filename,'wb')
  for t in cr.parsed_sents() :
    parsedfile.write(normalize_parsed(cut_tags(t)).pprint(margin=9999999)+'\n')
  parsedfile.close()

  return

def bankstring(tb) :
  if tb == pennbank:
    bs = "Penn"
  elif tb == tubank:
    bs = "TueBaDZ"
  elif tb == tigerbank:
    bs = "Tiger"
  elif tb == french_spmrl :
    bs = "SPMRL_French"
  else :
    bs = "Penn" #default for now
  return bs

#TODO: check if the appropriate sample has already been generated (and saved as file!) --- only generate samples if needed!s
def create_samples (tb, seed, sample_size, training_fraction =0.9) :
  random.seed(seed)
  print('Creating training sample..')
  #training_indices = sorted(random.sample(xrange(banksize), int(training_fraction*banksize)))
  all_trees = map(filter_tree, random.sample(tb.parsed_sents(), sample_size))
  training_sample = all_trees[:int(training_fraction*sample_size)]
  test_sample = all_trees[int(training_fraction*sample_size):]
  #training_sample = [filter_strict(tb.parsed_sents()[i]) for i in training_indices] # cut punctuation
  print('Creating test sample..')
  test_sample = all_trees[int(training_fraction*sample_size):]
  random.seed(seed) #reset the random-number generator
  all_tags = random.sample(tb.tagged(), sample_size)
  #test_sample = [filter_strict(tb.parsed_sents()[i]) for i in xrange(banksize) if i not in training_indices] # cut punctuation
  tagged_sample = all_tags[int(training_fraction*sample_size):]
  #tag_sample = [[(term,tag) for (term,tag) in sent if not ignore_punct_tag(tag)] for sent in tagged_sample] #tags without punctuation
  tag_sample = [[(term,cut_tag(tag)) for (term,tag) in sent if not ignore_tag(tag)] for sent in tagged_sample] #tags of test trees
  return (training_sample, test_sample, tag_sample)

def write_samples_batch(tb, seeds, sizes) :
  for size in sizes :
    for seed in seeds:
      train,test,tags = create_samples(tb, seed, size)
      train_file = open('input_data/train_'+ bankstring(tb) + '_' + str(seed) + '_'+ str(size) + '.pkl', 'wb')
      test_file = open('input_data/test_'+ bankstring(tb) + '_' + str(seed) + '_'+ str(size) + '.pkl', 'wb')
      tags_file = open('input_data/tags_'+ bankstring(tb) + '_' + str(seed) + '_'+ str(size) +'.pkl', 'wb')
      pickle.dump(train,train_file)
      pickle.dump(test,test_file)
      pickle.dump(tags,tags_file)
      train_file.close()
      test_file.close()
      tags_file.close()

def load_samples(tb, seed, size) :
  train_file = open('input_data/train_'+ bankstring(tb) + '_'+ str(seed)+ '_'+ str(size) + '.pkl', 'rb')
  test_file = open('input_data/test_'+ bankstring(tb) + '_' + str(seed)+ '_'+ str(size) + '.pkl', 'rb')
  tags_file = open('input_data/tags_'+ bankstring(tb) + '_' + str(seed)+ '_'+ str(size) +'.pkl', 'rb')
  train = pickle.load(train_file)
  test = pickle.load(test_file)
  tags = pickle.load(tags_file)
  return (train,test,tags)

# parse the development set of the smpmrl treebank supplied
def parse_dev_treebank(language="French", small_training=False) :
  treebank = "SPMRL_"+language.upper()
  # TODO: use (and/or extend) the create_samples, write_samples and load_samples functions (to deal with whole treebanks)
  # parse the development sets of the treebanks and print the eval-results from them
  training_tb = nltk.corpus.LazyCorpusLoader("SPMRL/"+language.upper()+"_SPMRL/gold/ptb/train", nltk.corpus.BracketParseCorpusReader, r'train.'+language+'.gold.ptb')
  dev_tb = nltk.corpus.LazyCorpusLoader("SPMRL/"+language.upper()+"_SPMRL/gold/ptb/dev", nltk.corpus.BracketParseCorpusReader, r'dev.'+language+'.gold.ptb')
  train = training_tb.parsed_sents()
  test = dev_tb.parsed_sents()
  all_tags = dev_tb.tagged()
  tags = [[(term,cut_tag(tag)) for (term,tag) in sent if not ignore_tag(tag)] for sent in all_tags] #tags of test trees
  size = len(train)
  parse_pcfg_treebank(treebank, 00, size, train, test, tags, unfold_dim=True, VMarkov=1)
  parse_pcfg_treebank(treebank, 00, size, train, test, tags, unfold_dim=True)
  parse_pcfg_treebank(treebank, 00, size, train, test, tags, unfold_dim=False, VMarkov=1)
  parse_pcfg_treebank(treebank, 00, size, train, test, tags, unfold_dim=True, VMarkov=1, HMarkov=2)
  parse_pcfg_treebank(treebank, 00, size, train, test, tags, unfold_dim=False)
  parse_pcfg_treebank(treebank, 00, size, train, test, tags, unfold_height=True)
  parse_pcfg_treebank(treebank, 00, size, train, test, tags, unfold_height=True, VMarkov=1)


#TODO: maybe store+load annotated training/tag-samples and do not generate them anew each time
def eval_pcfg_batch(tb,sizes,seeds) :
  for samp in sizes:
    for seed in seeds:
      
      train,test,tags = load_samples(tb,seed,samp)
      
      parse_pcfg_treebank(bankstring(tb), seed, samp, train, test, tags, unfold_dim=False)
      eval_parses(bankstring(tb), seed, samp, unfold_dim=False, eval_method = "evalb")
      eval_parses(bankstring(tb), seed, samp, unfold_dim=False, eval_method = "CB")
      eval_parses(bankstring(tb), seed, samp, unfold_dim=False, eval_method = "LA")

      parse_pcfg_treebank(bankstring(tb), seed, samp, train, test, tags, unfold_dim=True)
      eval_parses(bankstring(tb), seed, samp, unfold_dim=True, eval_method = "evalb")
      eval_parses(bankstring(tb), seed, samp, unfold_dim=True, eval_method = "CB")
      eval_parses(bankstring(tb), seed, samp, unfold_dim=True, eval_method = "LA")

      parse_pcfg_treebank(bankstring(tb), seed, samp, train, test, tags, unfold_dim=False, VMarkov=1)
      eval_parses(bankstring(tb), seed, samp, unfold_dim=False, VMarkov=1, eval_method = "evalb")
      eval_parses(bankstring(tb), seed, samp, unfold_dim=False, VMarkov=1, eval_method = "CB")
      eval_parses(bankstring(tb), seed, samp, unfold_dim=False, VMarkov=1, eval_method = "LA")

      parse_pcfg_treebank(bankstring(tb), seed, samp, train, test, tags, unfold_dim=True, VMarkov=1)
      eval_parses(bankstring(tb), seed, samp, unfold_dim=True, VMarkov=1, eval_method = "evalb")
      eval_parses(bankstring(tb), seed, samp, unfold_dim=True, VMarkov=1, eval_method = "CB")
      eval_parses(bankstring(tb), seed, samp, unfold_dim=True, VMarkov=1, eval_method = "LA")

      parse_pcfg_treebank(bankstring(tb), seed, samp, train, test, tags, unfold_height=True)
      eval_parses(bankstring(tb), seed, samp, unfold_height=True, eval_method = "evalb")
      eval_parses(bankstring(tb), seed, samp, unfold_height=True, eval_method = "CB")
      eval_parses(bankstring(tb), seed, samp, unfold_height=True, eval_method = "LA")

      parse_pcfg_treebank(bankstring(tb), seed, samp, train, test, tags, unfold_height=True, VMarkov=1)
      eval_parses(bankstring(tb), seed, samp, unfold_height=True, VMarkov=1, eval_method = "evalb")
      eval_parses(bankstring(tb), seed, samp, unfold_height=True, VMarkov=1, eval_method = "CB")
      eval_parses(bankstring(tb), seed, samp, unfold_height=True, VMarkov=1, eval_method = "LA")

#      parse_pcfg_treebank(tb, seed, samp, train, test, tags, unfold_dim=True, VMarkov=1, HMarkov=2)
#      eval_parses(tb, seed, samp, unfold_dim=True, VMarkov=1, HMarkov=2, eval_method = "evalb")
#      eval_parses(tb, seed, samp, unfold_dim=True, VMarkov=1, HMarkov=2, eval_method = "CB")
#      eval_parses(tb, seed, samp, unfold_dim=True, VMarkov=1, HMarkov=2, eval_method = "LA")


def eval_pcfg_single(tb, size) :
  train,test,tags = load_samples(tb,42,size)
  parse_pcfg_treebank(bankstring(tb), 42, size, train, test, tags, unfold_dim=False)
  parse_pcfg_treebank(bankstring(tb), 42, size, train, test, tags, unfold_dim=True)
  eval_parses(bankstring(tb), 42, size, unfold_dim=True, eval_method = "evalb")
  eval_parses(bankstring(tb), 42, size, unfold_dim=False, eval_method = "evalb")

def eacl_test() :
  #seeds for the python random number generator (fixed here for reproducibility!)---
  # these seeds were chosen by "for i in {1..10}; do od -An -N2 -i /dev/urandom; done" on the author's shell :)
  # For small sample sizes it is important to run our experiments with several different training/test-sets for cross-validation!
  # This ensures that our overall findings are neither statistical flukes nor due to overfitting parameters to the test-set
  # To get reliable data on the parsing speed this is crucial!
  seeds = [1271,49647,53316,25907,13023,898,30964,51253,54469,45047]
  # sizes = [5000,10000,20000,30000,40000,50000] #first batch
  sizes = [60000,70000] # second batch
  #write_samples_batch(tubank, seeds, sizes) #only have to call that once..
  eval_pcfg_batch(tubank,sizes,seeds)

def main(sizes) :
  #seeds = [1271,49647,53316,25907,13023,898,30964,51253,54469,45047]
  seeds = [42]
  eval_pcfg_batch(tubank, sizes, seeds)

if __name__ == "__main__":
  main(map(int,sys.argv[1:]))
