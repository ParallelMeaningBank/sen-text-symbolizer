# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 20:27:14 2016

@author: duy

Extract noun from WN
"""

import en;
import re;

def addtilde(inpstr):
    return re.sub("[^A-Za-z0-9.\-']+and[^A-Za-z0-9.\-']+|[^A-Za-z0-9.\-']+","~",inpstr);
    
if __name__ == "__main__":
    with open("wn_nouns2.csv","w+") as noun_file:
        for n in en.wordnet.wordnet.N:
            singular = re.sub("^[\W\d].*|\(.*$","",str(n)).strip();
            if len(singular)<2: continue;
            plural = en.noun.plural(singular)
            #noun_file.write("{0},{0},#CON,,{0},{1},#CON\n".format(singular, plural)); 
            noun_file.write("{0},{0},#CON,,{0},{1},#CON\n".format(addtilde(singular), addtilde(plural))); 