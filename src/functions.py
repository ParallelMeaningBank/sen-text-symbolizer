# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:51:22 2016
@project: SEN text symbolizer - Parallel meaning bank - RUG

@author: duy
"""
import sys,csv,re,random,pickle
import numpy as np
from collections import Counter
from difflib import SequenceMatcher as similar;
from pathlib import Path
import Datasets

MIN_SEQUENCE_MATCH = 0.8;   # Minimum percentage of match to unify 2 words


def wordExtraction(word):
    counter = Counter(re.sub('([\s+A-Z~]|and)', '', word)) # Remove Upper-cased & spec
    vector = np.zeros(26,np.uint8); # Create a vector with int8 values stores
                     # freq of chars in word. 26 is size of English alphabet
    a_pos= ord("a")  # ASCII position of "a", beginning in alphabet
    for c in counter: vector[ord(c)-a_pos]=counter[c];
    return vector 

def numberStringExtract(nstr):
    tokens = nstr.split('~');               # Split the string
    vectors=[]      # Store extracted vector of all tokens
    max_match = 0 # value of the token that is most likely to "hundred"
    max_pos = -1                   # Position of this smalest distance
    index = -1
    for tk in tokens:
        index +=1;
        vtr = wordExtraction(tk); # Extract the word
        vectors.append(vtr)
        similarity= similar(None, tk, "hundred").ratio()
        if ( similarity>max_match and similarity>MIN_SEQUENCE_MATCH):
            max_pos = index;    # Save the position of the token
    if (max_pos == -1): # No hundred found
        # Just sum all of them, and add a zero vector behind
        sum_vec = np.sum(vectors,axis=0);
        return  np.hstack((np.zeros(26,np.uint8),sum_vec));
    else:
        hundres_sum_vecs = np.sum(vectors[:max_pos],axis=0);
        #print hundres_sum_vecs;
        unit_sum_vecs = np.zeros(26,np.uint8);
        #print unit_sum_vecs;
        #print max_pos<(len(tokens)-1)        
        if (max_pos<(len(tokens)-1)): # "hundred" is NOT at the end of sequence        
           unit_sum_vecs = np.sum(vectors[(max_pos+1):],axis=0); 
        return np.hstack((hundres_sum_vecs,unit_sum_vecs));
            

        
