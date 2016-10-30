# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:12:07 2016

@author: duy

Text extraction for NN

@input: CSV file that contains num->text
@output: CSV file that contains label \t vector

@param: csv_file: the input file
"""
import sys,csv,re,random,pickle
import numpy as np
from collections import Counter
from difflib import SequenceMatcher as similar;
from pathlib import Path
import Datasets,functions

TEST_RATE=0.3               # Rate of text, out of full data size

def main(_):
    print '\!'

# Count occurence of letters in a sequence. It return an array with index from
# 0-26, which is English alphabet  
###############################################################################
def LoadDataset(infile_path):
    
    if Path(infile_path).is_file():
        with open(infile_path,'rt') as fresh_csv_file:
            print "Start loading data from ",infile_path
            rcount = 0;
            full_txt = np.empty(shape=[0,52],dtype=np.uint8) # No row, 52 is dims of vect
            full_lb = np.empty(shape=[0,1000],dtype=np.uint8)
            try:
                reader = csv.reader(fresh_csv_file,delimiter=',')
                for row in reader:
                    if (rcount!=0):
                        #print row
                        full_txt = np.vstack((full_txt,functions.numberStringExtract(row[1]))) 
                        lbvr = np.zeros(1000,np.int8); # Initialize empty label
                        lbvr[int(row[0])]=1;            # Assign value
                        full_lb = np.vstack((full_lb,lbvr))
                        #print full_txt, full_lb
                        if rcount%100 ==0:
                            print "-> Extracted ", rcount ," records"
                    rcount +=1
                print "Loaded datafile. VectorShape: ", full_txt.shape
                print "                 LabelSuape: ",full_lb.shape
            except ValueError as e:
                print "Oops!  That was no valid row.  Try again..."
                print e
            finally:
                fresh_csv_file.close()
            print rcount;

            ####### Now split the data into train & test
            #Create a list of random sample 
            n_row= full_txt.shape[0];
            n_test = int(n_row * TEST_RATE)
            test_indeces = random.sample(range(n_row),n_test) 
            #print test_indeces
            # Now extract from the full dataset, to form train and test set            
#            test_vectors = full_txt; # Get the test vectors
#            test_labels = full_lb; # Get the test labels
            test_vectors = full_txt[test_indeces,] # Get the test vectors
            test_labels =  full_lb[test_indeces,] # Get the test labels
 
            train_vectors = np.delete(full_txt,test_indeces,0)# Remove items in
                    # test set, this will end up with a train set
            train_labels = np.delete(full_lb,test_indeces,0) # Remove items in
                    # test set, this will end up with a train set        
        data_sets = Datasets.DataSets()
        # Validate the matrix size
        if ((np.sum((train_vectors.shape,test_vectors.shape),1)==full_txt.shape).all() 
            and (np.sum((train_labels.shape,test_labels.shape),1).all()==full_lb.shape)).all():
            print "Train/test split validated"
        data_sets.train = Datasets.DataSet(train_vectors, train_labels)
        data_sets.test = Datasets.DataSet(test_vectors, test_labels)
        return data_sets
        
if __name__=='__main__':
    # Read fresh CSV file
    if (sys.argv[1]!=''):
        text_datasets = LoadDataset(sys.argv[1])
        print "Train dataset shape: ",text_datasets.train.texts.shape, " - " , text_datasets.train.labels.shape
        print "Test dataset shape: ",text_datasets.test.texts.shape, " - " , text_datasets.test.labels.shape
        # Now save the file in pickle format
        pickle.dump( text_datasets, open( re.search("(.+?)(\.[^.]*$|$)",sys.argv[1]).group(0)+ ".p", "wb" ))