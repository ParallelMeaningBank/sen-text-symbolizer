# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:51:22 2016
@project: SEN text symbolizer - Parallel meaning bank - RUG

@author: duy

This file contains common functions for the projects
"""
import sys,csv,re,random,pickle
import numpy as np
from collections import Counter
from difflib import SequenceMatcher as similar;
from pathlib import Path
import Datasets, SpelledNumber,math

MIN_SEQUENCE_MATCH = 0.8;   # Minimum percentage of match to unify 2 words

"""
Extract feature vector from a word
"""
def wordExtraction(word):
    # Remove Upper-cased & spec
    counter = Counter(re.sub('([\s+A-Z~]|and)', '', word)) 
    vector = np.zeros(26,np.uint8); # Create a vector with int8 values stores
                     # freq of chars in word. 26 is size of English alphabet
    a_pos= ord("a")  # ASCII position of "a", beginning in alphabet
    for c in counter: 
        i = ord(c)-a_pos;        
        if (i>=0): vector[i]=counter[c];
    return vector 

###############################################################################

"""
Extract feature vector from a seuqnce of words
"""
def numberStringExtract(instr):
    tokens = instr.split('~');               # Split the string
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

###############################################################################

"""
Fetching a sequence of words into subsequence, corresponding to anchors. Resul-
ting in an dictionary, whereby:
* out.units: units  (0*10^0 to 999*10^0)
* out.thousands: thousands (0*10^3 to 999*10^3)
* out.millions: millions (0*10^6 to 999*10^6)
* out.billions: billions (0*10^9 to 999*10^9)
* out.trillions: trillions (0*10^12 to 999*10^12)
"""

def sequenceFetch(instr):
    # initialize the output
    res = { 
        'unit' : "",
        'thousand' : "",
        'million' : "",
        'billion' : "",
        'trillion' : ""
    }    
    # normalize the string 
    temp = re.sub("[^a-z0-9~.]+","",instr.lower());
    tokens = temp.split('~'); # Slit to tokens
    # find the anchor with certain  level of MIN_SEQUENCE_MATCH 
    anchors = ['trillion','billion','million','thousand'];
    
    for anchor in anchors: # Go throught all anchors
        #print(anchor, " ++++ ", tokens )
        # find the anchor in the list of token
        if (not tokens): break; # Nothing left on the string
        
        best_positions = -1;
        # Ensure the similarity is no less than this thredhold (+10% stricter)
        best_similarity = MIN_SEQUENCE_MATCH+0.1;
        
        for i in xrange(0,len(tokens)):
            similarity = similar(None, tokens[i], anchor).ratio();
            if (similarity > best_similarity):
                best_positions = i;
                best_similarity = similarity;
        
        if (best_positions==-1): # Anchor not found
            #index+=1;
            continue; # Passby
        else:
            seq = '~'.join(tokens[:best_positions]);
            if (not seq or seq=="a"): 
                seq='one'; # solve cases: thousands of, a thousand of -> 1,000
            elif seq=="half~a":
                seq='0.5';
            res[anchor] = seq;
            #print (anchor,"-----",'~'.join(tokens[:best_positions]))
            tokens = tokens[best_positions+1:]; # Eliminate recognized sequence
            
    if (tokens): res['unit'] = '~'.join(tokens); # last is units
    #print(res['trillion'],res['billion'],res['million'],res['thousand'],
    #      res['unit'])
    return SpelledNumber.SpelledNumber(res['trillion'],res['billion'],
                                       res['million'],res['thousand'],
                                       res['unit']);

###############################################################################




