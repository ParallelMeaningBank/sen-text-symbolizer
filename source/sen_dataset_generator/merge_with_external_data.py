# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 21:38:50 2017

@author: Nguyen Duc Duy
PMB
This combines a artificial generated file with a external file. The artitifical
file should have space separated between character (like: "a ~ t e x t"). The
external file, in another hand, can be space separated or not. However, files
without space-separated contents must have "_raw" ending in name (e.g filename.
train.sym_raw, filename.test.word_raw). The words and their symbols must be sto
red in separated file, ending with .word and .sym respectively.

Important: The input must come in the same file name prefix. The program will 
look for ending pattern (.train.sym, train.word, dev.sym...) to identify file 
content. 

@param:
 - input ARTIFICIAL file prefix
 - input EXTERNAL file prefix
 - output COMBINED file prefix
@usage
python merge_with_external_data [art_prefix] [ext_prefix] [out_prefix]

e.g:
python merge_with_external_data.py numtimegpoevetamdacdlcu-text_0.1_0.1_all-ws pmb_2016-12-14_has_at_least_one_semorsymbow_filtered combined_data
"""
import random,sys,re,codecs,os.path;
import numpy as np;

###############################################################################
def addtilde(inpstr):
    return re.sub("[^A-Za-z0-9.\-]+and[^A-Za-z0-9.\-]+|[^A-Za-z0-9.\-]+","~",inpstr);

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

def get_artificial_files(art_path_root):
    art_train_word = None; 
    art_test_word = None;
    art_dev_word = None;
    art_train_sym = None; 
    art_test_sym = None;
    art_dev_sym = None;
    
    # Check symbol files
    #print("{0}.train.sym".format(art_path_root))
    if os.path.exists("{0}.train.sym".format(art_path_root)):
        art_train_sym = "{0}.train.sym".format(art_path_root);
    if os.path.exists("{0}.test.sym".format(art_path_root)):
        art_test_sym = "{0}.test.sym".format(art_path_root);
    if os.path.exists("{0}.dev.sym".format(art_path_root)):
        art_dev_sym = "{0}.dev.sym".format(art_path_root);
        
    #Check word files
    if os.path.exists("{0}.train.word".format(art_path_root)):
        art_train_word = "{0}.train.word".format(art_path_root);
    if os.path.exists("{0}.test.word".format(art_path_root)):
        art_test_word = "{0}.test.word".format(art_path_root);
    if os.path.exists("{0}.dev.word".format(art_path_root)):
        art_dev_word = "{0}.dev.word".format(art_path_root);
        
    if art_train_word and art_test_word and art_dev_word \
                      and art_train_sym and art_test_sym and art_dev_sym:
        return art_train_word,art_test_word,art_dev_word,\
                art_train_sym,art_test_sym,art_dev_sym;
    else:
        print("Artificial file(s) not found.")
        print art_train_word,art_test_word,art_dev_word,\
                art_train_sym,art_test_sym,art_dev_sym;
        sys.exit(-1)

###############################################################################
        
def get_external_files(ext_path_root):
    ext_train_word = None; 
    ext_test_word = None;
    ext_dev_word = None;
    ext_train_sym = None; 
    ext_test_sym = None;
    ext_dev_sym = None;
    
    # Check symbol files
    if os.path.exists("{0}.train.sym".format(ext_path_root)):
        ext_train_sym = "{0}.train.sym".format(ext_path_root);
    if os.path.exists("{0}.test.sym".format(ext_path_root)):
        ext_test_sym = "{0}.test.sym".format(ext_path_root);
    if os.path.exists("{0}.dev.sym".format(ext_path_root)):
        ext_dev_sym = "{0}.dev.sym".format(ext_path_root);
        
    #Check word files
    if os.path.exists("{0}.train.word".format(ext_path_root)):
        ext_train_word = "{0}.train.word".format(ext_path_root);
    if os.path.exists("{0}.test.word".format(ext_path_root)):
        ext_test_word = "{0}.test.word".format(ext_path_root);
    if os.path.exists("{0}.dev.word".format(ext_path_root)):
        ext_dev_word = "{0}.dev.word".format(ext_path_root);
        
    if ext_train_word and ext_test_word and ext_dev_word \
                      and ext_train_sym and ext_test_sym and ext_dev_sym:
        print("Found all external files!")
        return ext_train_word,ext_test_word,ext_dev_word,\
                ext_train_sym,ext_test_sym,ext_dev_sym;
    else:
        print("Separated external train,test,dev file(s) not found. Now searching for joined file.")
        #sys.exit(-1)
    # Now search for all in one file
    ext_AIO_word = None; 
    ext_AIO_sym = None;
    raw_flag = True; # True if the file is raw (not space separated letters).
                    #  True: "hapiness" False: "h a p p i n e s s"
    if os.path.exists("{0}.sym".format(ext_path_root)):
        raw_flag = False
        ext_AIO_sym = "{0}.sym".format(ext_path_root);
    if os.path.exists("{0}.word".format(ext_path_root)):
        raw_flag = False
        ext_AIO_word = "{0}.word".format(ext_path_root);
    
    if os.path.exists("{0}.sym_raw".format(ext_path_root)):
        ext_AIO_sym = "{0}.sym_raw".format(ext_path_root);
    if os.path.exists("{0}.word_raw".format(ext_path_root)):
        ext_AIO_word = "{0}.word_raw".format(ext_path_root);  
        
    # Split the file if it is AIO
    if ext_AIO_word and ext_AIO_sym: # Exists AIO files
     # Read external sym
        external_sym_f =  codecs.open(ext_AIO_sym, encoding='utf-8', mode='r');
        if raw_flag:
            external_syms = [addspaces(line.strip("\n"),1) for line in external_sym_f if line!=u""];
        else:
            external_syms = [line.strip("\n") for line in external_sym_f if line!=u""];
            
        print("Loaded external sym! size: {0}. Raw mode: {1}".format(len(external_syms),raw_flag));
        external_sym_f.close();
        
        # Read external word
        external_word = [];
        external_word_f =  codecs.open(ext_AIO_word, encoding='utf-8', mode='r');
        for line in external_word_f:
            if line!=u"":
                items = line.strip("\n").split("	");
                if (len(items)==2):
                    tmp = items[0] + u" ~ " + addspaces(items[1],1)
                external_word.append(tmp);
        print("Loaded external word! size: {0}".format(len(external_word)));
        external_word_f.close();
        
        if len(external_word) != len(external_syms):
            print("[E] Fatal error. Mismatch external_word and external_word size.")
            sys.exit()
        external_size = len(external_word);
        # Now split artificial data into 3 list
        external_train_sym = [];
        external_train_word = [];
        external_dev_sym = [];
        external_dev_word = [];
        external_test_sym = [];
        external_test_word = [];
        for i in xrange(external_size):
            file_index = np.random.choice(a=[0,1,2],p=[0.7,0.2,0.1]);
            if (file_index==0): # Train item
                external_train_sym.append(external_syms[i]);
                external_train_word.append(external_word[i]);
            elif (file_index==1): #Dev item
                external_dev_sym.append(external_syms[i]);
                external_dev_word.append(external_word[i]);
            else: # test item
                external_test_sym.append(external_syms[i]);
                external_test_word.append(external_word[i]);
        
        # Now write them to file
        with codecs.open("{0}.train.sym".format(ext_path_root),encoding='utf-8', mode='w') as external_train_sym_f:
            for sym in external_train_sym:
                external_train_sym_f.write(sym+u"\n");
        
        with codecs.open("{0}.train.word".format(ext_path_root),encoding='utf-8', mode='w') as external_train_word_f:
            for word in external_train_word:
                external_train_word_f.write(word+u"\n");
                
        with codecs.open("{0}.dev.sym".format(ext_path_root),encoding='utf-8', mode='w') as external_dev_sym_f:
            for sym in external_dev_sym:
                external_dev_sym_f.write(sym+u"\n");
        
        with codecs.open("{0}.dev.word".format(ext_path_root),encoding='utf-8', mode='w') as external_dev_word_f:
            for word in external_dev_word:
                external_dev_word_f.write(word+u"\n");
                
        with codecs.open("{0}.test.sym".format(ext_path_root),encoding='utf-8', mode='w') as external_test_sym_f:
            for sym in external_test_sym:
                external_test_sym_f.write(sym+u"\n");
        
        with codecs.open("{0}.test.word".format(ext_path_root),encoding='utf-8', mode='w') as external_test_word_f:
            for word in external_test_word:
                external_test_word_f.write(word+u"\n");
         
        print("External files proccessing completed!Now validated file existence.");
        # Now call back  to verity file exists
        return get_external_files(ext_path_root);
    else:
        print("[E] Error when processing the external file!")
        sys.exit(-1)
        
        
###############################################################################

if __name__ =='__main__':
    """ This program join artificial generated and external source file to make
    train, dev and test set. Notice that the input artificial data consist of 3
    separated set train,dev,test set. External source data, in another hand, are
    together in one file. So we randomly read it into 3 parts and join separatedly
    to artificial data sets"""
    
    artificial_files_prefix = sys.argv[1];
    external_files_prefix = sys.argv[2];
    output_files_prefix = sys.argv[3];
    
    art_train_word_p,art_test_word_p,art_dev_word_p,art_train_sym_p,art_test_sym_p,art_dev_sym_p = get_artificial_files(artificial_files_prefix)
    
    
    ext_train_word_p,ext_test_word_p,ext_dev_word_p,ext_train_sym_p,ext_test_sym_p,ext_dev_sym_p = get_external_files(external_files_prefix)

    
    # First check file existents
    
       
    # Read artificial data, merge them into final train, dev files
    #   Development set = Artificial development set + external development set
    #   Training set = Artificial training set + external training set
    #   Test sets are not joined. The artificial test set are generated prior,
    #       the external test set is stored at the same directory as original 
    #       external data.
     
    # READ ARTIFICIAL FILES
    # Read artificial train file
    art_train_sym_f =  codecs.open(art_train_sym_p, encoding='utf-8', mode='r');
    art_train_syms = [line.strip("\n") for line in art_train_sym_f if line];
    print("Loaded ARTIFICIAL train sym! size: {0}".format(len(art_train_syms)));
    art_train_sym_f.close();
    
    art_train_word_f =  codecs.open(art_train_word_p, encoding='utf-8', mode='r');
    art_train_word = [line.strip("\n") for line in art_train_word_f if line];
    print("Loaded ARTIFICIAL train word! size: {0}".format(len(art_train_word)))
    art_train_word_f.close();


    # Read artificial dev file
    art_dev_sym_f =  codecs.open(art_dev_sym_p, encoding='utf-8', mode='r');
    art_dev_syms = [line.strip("\n") for line in art_dev_sym_f if line];
    print("Loaded ARTIFICIAL dev sym! size: {0}".format(len(art_dev_syms)));
    art_dev_sym_f.close();
    
    art_dev_word_f =  codecs.open(art_dev_word_p, encoding='utf-8', mode='r');
    art_dev_word = [line.strip("\n") for line in art_dev_word_f if line];
    print("Loaded ARTIFICIAL dev word! size: {0}".format(len(art_dev_word)))
    art_dev_word_f.close();
    
    # READ EXTERNAL FILES
    # Read EXTERNAL train file
    ext_train_sym_f =  codecs.open(ext_train_sym_p, encoding='utf-8', mode='r');
    ext_train_syms = [line.strip("\n") for line in ext_train_sym_f if line];
    print("Loaded EXTERNAL train sym! size: {0}".format(len(ext_train_syms)));
    ext_train_sym_f.close();
    
    ext_train_word_f =  codecs.open(ext_train_word_p, encoding='utf-8', mode='r');
    ext_train_word = [line.strip("\n") for line in ext_train_word_f if line];
    print("Loaded EXTERNAL train word! train size: {0}".format(len(ext_train_word)))
    ext_train_word_f.close();


    # Read EXTERNAL dev file
    ext_dev_sym_f =  codecs.open(ext_dev_sym_p, encoding='utf-8', mode='r');
    ext_dev_syms = [line.strip("\n") for line in ext_dev_sym_f if line];
    print("Loaded EXTERNAL dev sym! size: {0}".format(len(ext_dev_syms)));
    ext_dev_sym_f.close();
    
    ext_dev_word_f =  codecs.open(ext_dev_word_p, encoding='utf-8', mode='r');
    ext_dev_word = [line.strip("\n") for line in ext_dev_word_f if line];
    print("Loaded EXTERNAL dev word! size: {0}".format(len(ext_dev_word)))
    ext_dev_word_f.close();
    
    
    # BUILD FINAL TRAIN SET
    full_trainset_sym = ext_train_syms;
    full_trainset_sym.extend(art_train_syms);     
    full_trainset_word = ext_train_word;
    full_trainset_word.extend(art_train_word)
    full_trainset_len = len(full_trainset_sym);
    print("1. Combined train set size:" + str(full_trainset_len))
    full_trainset_sym_final = []
    full_trainset_word_final = []
    

    if full_trainset_len!=len(full_trainset_word):
        print("[E] Mismatch full_trainset_sym and full_trainset_word size.")
        sys.exit()
    else:
        # Do the suffle
        random_indexs = random.sample(range(full_trainset_len),full_trainset_len);
        for index in random_indexs:
            full_trainset_sym_final.append(full_trainset_sym[index]);
            full_trainset_word_final.append(full_trainset_word[index]);
        # the releae memory
        full_trainset_sym= None;
        full_trainset_word= None;
        # write to file
        with codecs.open(output_files_prefix + ".train.sym", encoding='utf-8',\
                        mode='w') as final_train_sym_f:
            for sym in full_trainset_sym_final:
                final_train_sym_f.write(sym+u"\n");
        with codecs.open(output_files_prefix + ".train.word",encoding='utf-8',\
                        mode='w') as final_train_word_f:
            for word in full_trainset_word_final:
                final_train_word_f.write(word+u"\n");
        with codecs.open(output_files_prefix + ".train.csv", encoding='utf-8',\
                mode='w') as final_train_csv_f:
            for i in xrange(len(full_trainset_word_final)):
                tmp = u"\"{0}\",\"{1}\"\n".format(full_trainset_sym_final[i],\
                                                full_trainset_word_final[i]);
                final_train_csv_f.write(tmp);
            
     # Now build the FINAL dev set
    full_devset_sym = ext_dev_syms; full_devset_sym.extend(art_dev_syms);     
    full_devset_word = ext_dev_word; full_devset_word.extend(art_dev_word)
    full_devset_len = len(full_devset_sym);
    print("2. Combined dev set size:" + str(full_devset_len))
    full_devset_sym_final = []
    full_devset_word_final = []
    if full_devset_len!=len(full_devset_word):
        print("[E] Mismatch full_devset_sym and full_devset_word size.")
        sys.exit()
    else:
        # Do the suffle
        random_indexs = random.sample(range(full_devset_len), full_devset_len);
        for index in random_indexs:
            full_devset_sym_final.append(full_devset_sym[index]);
            full_devset_word_final.append(full_devset_word[index]);
        # the releae memory
        full_devset_sym= None;
        full_devset_word= None;
        # write to file
        with codecs.open(output_files_prefix + ".dev.sym", encoding='utf-8',\
                        mode='w') as final_dev_sym_f:
            for sym in full_devset_sym_final:
                final_dev_sym_f.write(sym+u"\n");
        with codecs.open(output_files_prefix + ".dev.word", encoding='utf-8',\
                        mode='w') as final_dev_word_f:
            for word in full_devset_word_final:
                final_dev_word_f.write(word+u"\n");
        with codecs.open(output_files_prefix + ".dev.csv", encoding='utf-8',\
                mode='w') as final_dev_csv_f:
            for i in xrange(len(full_devset_word_final)):
                tmp = u"\"{0}\",\"{1}\"\n".format(full_devset_sym_final[i],\
                                                  full_devset_word_final[i]);
                final_dev_csv_f.write(tmp);        
    print("Finish building trainset (size: {0},{1}) & devset (size: {1},{3})"\
        .format(len(full_trainset_sym_final),\
                len(full_trainset_word_final),\
                len(full_devset_sym_final),\
                len(full_devset_word_final)))