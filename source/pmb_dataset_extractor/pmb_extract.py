# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:17:05 2016

@author: Duc-Duy Nguyen
Extract documents with BOW from PMB for training and tesing purpose

"""

import os,sys,re,codecs,traceback;
import xml.etree.ElementTree as ET

class Record(object):
    def __init__(self, sym, tok, sem, loc):
        self._sym = sym; # Symbol
        self._tok = tok; # Original token lowercased
        self._sem = sem;
        self._loc = loc;
        
    def sym(self):
        return self._syn;
        
    def tok(self):
        return self._token;
        
    def sem(self):
        return self._sem;
        
    def loc(self):
        return self._loc;

out_file = None;
mannual_file = None;

def addtilde(inpstr):
    return re.sub("[^A-Za-z0-9.\-]+and[^A-Za-z0-9.\-]+|[^A-Za-z0-9.\-]+","~",inpstr);
    
def travelDir(rootDir):
    #print('\t%s' % rootDir)
    for dirName, subdirList, fileList in os.walk(rootDir, topdown=False):
        #print('Found directory: %s' % dirName)
        #langs = set([]);
        num_en_lemma_bows=0;
        sym_with_bow = []; # element is s[start_point,end_point,sym,token,semtag]
        num_en_sem_bows=0;
        #sem_with_bow = [];
        try:
            with open(dirName+'/en.lemma.bows') as lbf:
                for line in lbf:
                    num_en_lemma_bows+=1;
                    details = line.strip("\n").split("\t");
                    #print(details)
                    start_point = details[4];
                    end_point = details[5];
                    sym = details[6];
                    sym_with_bow.append([start_point,end_point,sym,None,None]) #No token, semtag yet
                    
            with open(dirName+'/en.semtag.bows') as sbf:
                num_en_sem_bows = sum(1 for _ in sbf)
        except Exception:
            print("Unable to count bow files number of line on file " + dirName)
            traceback.print_exc()
            continue
        
        if num_en_lemma_bows>0 or num_en_sem_bows>0:
            # Open lemma file
            try:
              
                lems = [];toks= []; sems= [];
                root= ET.parse(dirName+'/en.der.xml').getroot()
                tags_info = []
                for n in root.iter("lex"):
                    tok = n.find("token").text;
                    vals=None;
                    if tok!=u"ø":
                        vals = [tgs.text for tgs in n.findall("tag")]
                        toks.append(tok.lower())
                        sems.append(vals[0])
                        lems.append(vals[1])
                        tags_info.append(vals)
                    if vals:
                        for i in xrange(len(sym_with_bow)):
                            if sym_with_bow[i][0] == vals[3] and sym_with_bow[i][1] == vals[4]:
                                sym_with_bow[i][3] = tok;
                                sym_with_bow[i][4] = vals[0];
                #Write raw SYM file
                if len(lems) == len(toks) and len(toks) == len(sems):
                    for i in xrange(len(lems)):
                        if lems[i]!="":
                            tmp = u"\"{0}\",\"{1}\",\"#{2}\",\"{3}\"\n".format(lems[i],toks[i],sems[i],dirName);
                            out_file.write(tmp);
                else:
                    print("Mismatch sizes between lemma, token and semantic tag at doc {0}: {1} -- {2} -- {3}".format(dirName,lems, toks, sems))
                    continue;
                    
                #Write manual edited SYM file
                if sym_with_bow:
                    print(str(sym_with_bow)+ " @ " + dirName);
                for _,_,sym,token,semtag in sym_with_bow:
                    tmp = u"\"{0}\",\"{1}\",\"#{2}\",\"{3}\"\n".format(sym,token,semtag,dirName);
                    mannual_file.write(tmp);
                    
            except:
                print("[E] Unable to extract file from:" + dirName)
                traceback.print_exc()
if __name__=='__main__':
    out_file = codecs.open(sys.argv[2], encoding='utf-8', mode='w');
    mannual_file = codecs.open(sys.argv[2][:-3]+"man.csv", encoding='utf-8', mode='w');
    if out_file:
        with open(sys.argv[1],"rb") as accepted_f: # Open accepted open file
            # Now extract csv line by line, each line have format: "partID","docID"
            count = 0;
            for line in accepted_f:
                line = line.strip();
                #print(line)
                travelDir(line);
                count+=1;
                if count % 1000 ==0:
                    print("Extracted {0} lines".format(count))
    out_file.close();
