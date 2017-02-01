#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:45:36 2016

@author: Duc-Duy Nguyen
@ Parallel Meaning Bank

Generate number on range.

@input:
* 0. ZERO argument: reserved for script name
* 1. FIRST argument: sys.argv[1] MODE:
**   0: flankly number (0,1,2,3...)
**   1: Number and years (2016=twenty~sixteen,1998=nineteen~ninety~eight)
**   2: number and rank  (1st,2nd,3rd...)
**   3: all these above items (number, rankm years)
**   4: time (9am,9:00 a.m., haft past 6)
**   5: all these above items (number, rankm years,time)
**   6: all these above items and GPO
**   7: all these above items and EVE: EXS, ENS, EPS, EXG,ENG, EPG, EFG, EXT, 
**                                     ENT, EPT, EFT (EFS dosen't exist in EN)
**   8: all the items aboved, and TNS: NOW,PST,FUT,PFT. (PRG dosen't exist in EN)
**                            and ACT
**                            and MOD
**                            and DEM: PRX,DST (MED dosen't exist in EN)
**
**   9: all the items aboved, and ANA: PRO, DEF, HAS, REF, EMP
**                            and COM: EQA, MOR, LES, TOP, BOT, ORD
**                            and DIS
**                            and LOG
**   10: all the items aboved, and CON extracted from wordnet
**   11: TNS, MOD, ACT, DEM, ANA, COM, DIS, LOG, TTL, EVE, CON, UOM, QUC, SCO, DOW, YOC, GPO
**
* 2. SECOND argument: sys.argv[2] SIZE 
*    var: size - is number of samples (default is 30k)
* 3. THIRD argument: sys.argv[3] OVERALL ERROR RATE
*    var: overall_noise - overall noise rate (0.0-1.0)
* 4. FORTH argument: sys.argv[4] SAMPLE ERROR RATE
*    var: sample_noise - sample noise rate (0.0-1.0)
* 5. FIFTH argument: sys.argv[5] SPACE-SEPARATED
*    var: spc if 1 then enable. default is 0. E.g:"twenty~one" -> "t w e n t y ~ o n e"
* 6. SIXTH argument: sys.argv[6] OUTPUT FILE
*    var: out_file: the output file in CSV
* 7 FORWARD: depend on mode
*
* [Mode 0 to 3] num, ordinal, year
**   sys.argv[7]: var start_point - the low value. Default is 0
**   sys.argv[8]: var end_point: the high. Default is 999
**   sys.argv[9]: distribution_mode ("exponential","uniform","gaussian")
* [Mode 4] Time only
**   no further argument
* [Mode 5] num, ordinal, year, time
**   sys.argv[7]: var start_point - the low value. Default is 0
**   sys.argv[8]: var end_point: the high. Default is 999
* 1. FIRST argument: sys.argv[1] MODE:
USAGE:
python dataset_generate.py 3 50000 0.1 0.1 1 out.csv 0 999999 exponential
"""
from __future__ import division;
import sys,random,re,math,codecs,en,roman,threading;
import inflect;
import numpy as np;
#import matplotlib.pyplot as plt
from datetime import time

from data_utils import gpo_list,english_infinitive_verbs,day_of_week, \
                        month_of_year,tns_list,mod_list,dem_list, \
                        act_list,com_list,ana_list,dis_list,log_list,uom_list,\
                        ttl_list

NUMBER_OF_GENERATOR=80;

start_point = 0;
end_point = 999999;
dataset_size=30000;
mode=1;
ovr_err=0.2;
smp_err=0.2;
spc = 0;
reduce_rate = 0.00001; # Prob perform random reduce. E.g: 2245 -> thousands
distribution_mode = "exponential";

rand_values=list();
conList=None; # Init empty concept list
eveList = None;

err_choices = "abcdefghijklmnopqrstuvwxyz";

time_formats = ["%H:%M"  ,"%-H:%M"  ,                       # index 0-1 is 12:30,1:30,
                "%I:%M%p","%-I:%M%p","%I:%M~%p","%-I:%M~%p",# index 2-5 is 09:15PM, 9:15PM, 06:15~PM, 9:15~PM
                "%I%p","%-I%p","%I~%p","%-I~%p"             # index 6-9 is 01:00PM, 1PM, 01~PM, 1~PM
                ]
fixed_time_terms = [("00:00","midnight"),
                    ("12:00","midday"),
                    ("12:00","noon"), 
                    ("12:00","noontide"), 
                    ("12:00","noontime")]

# Distribution of the time_formats 
time_format_probabilities = [ 0.06, 0.05,                # index 0-2     0.11
                              0.05, 0.05,  0.05, 0.05,   # index 2-5     0.2
                              0.05, 0.04,  0.03, 0.05,   # index 6-9     0.17
                              0.04, 0.055, 0.04, 0.055,  # index 10-13   0.19
                              0.04, 0.055, 0.04, 0.055,  # index 14-17   0.19
                              0.02, 0.055, 0.065         # index 18-20   0.14
                            ] # sum = 1.0

toolbar_width = 100
toolbar_value=0;

##############################################################################
def addtilde(inpstr):
    return re.sub("[^A-Za-z0-9.\-]+and[^A-Za-z0-9.\-]+|[^A-Za-z0-9.\-]+","~",inpstr);
    
##############################################################################
    
def build_concept_list(conlist = []):
    rs = [];
    for n in conlist:
        singular = re.sub("^[\W\d].*|\(.*$","",str(n)).strip();
        if len(singular)<3 or len(singular) >30: continue;
        plural = addtilde(en.noun.plural(singular))
        singular = addtilde(singular);
        
        #noun_file.write("{0},{0},#CON,,{0},{1},#CON\n".format(singular, plural)); 
        rs.append((singular,singular,"#CON")); 
        rs.append((singular,plural,"#CON"));
    return rs;
        
##############################################################################

def addnoise(inpstr,rate): 
    rs = "";
    if rate>0:
        # Compute number of char that contains error
        n_err = int(len(inpstr)*rate);
        #z n_err;
        # Randomly select characters that contains error
        error_charls= set([]);
        while len(error_charls)<n_err:
            i =np.random.randint(0,len(inpstr)-1);
            if ((i not in error_charls) and inpstr[i]!='~'):
                error_charls.add(i);
        error_charls = sorted(error_charls);
        #print inpstr, " | ", n_err, " | ", error_charls;
        
        # Now create errors
        for i in xrange(0,len(inpstr)):
            if (error_charls): # the list is not empty
                if (i< error_charls[0]):
                   rs +=  inpstr[i];
                elif (i==error_charls[0]):
                    #randomly select an error type
                    err_type =np.random.randint(0,3);
                    if (err_type==0): # Replace by another character
                        rs+=random.choice(err_choices);
                    elif (err_type==1): # insert a character 
                        rs+=inpstr[i]+random.choice(err_choices);
                    #else : missing a charcter, doing nothing
                    error_charls.remove(error_charls[0]);
            else:
                rs+=inpstr[i];
        #print " -- ", rs;
                
    return rs;

##############################################################################

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

        
##############################################################################

def convert_to_words(num):
    if (num<100): # ignore small number
        return num,p.number_to_words(num);
    # Decide whether the sample is reduced 
    if (np.random.randint(0,2)==1): # decided to be reduced
        if (num<1000): #number from 0-999
            hundred_val = truncate_float(num/100,1);
            # Check if it is integer, or randomly picked to be convert to int
            if (hundred_val.is_integer()):
                return int(hundred_val*100),"{0} hundred".format(int(hundred_val))
            else: #Float
                # Check if the last digit is 5 (mean a haft)
                frac,whole = math.modf(hundred_val); # Get dicimal
                if frac == 0.5:
                    # Decide to turn to a half or not
                    if (np.random.randint(0,2)==1):
                        return int(whole*100),"{0} hundred and a half".format(int(whole))
                # Otherwise, just keep the original        
                return int(hundred_val*100),"{0} hundred".format(hundred_val)
        if (num<1000000): # number from 0-999999
            thousand_val = truncate_float(num/1000,1);
            exp_type = np.random.choice(["thousand","million","billion","trillion"],p=[0.15,0.35,0.35,0.15])
            exp_val = 1000 if exp_type == "thousand" else 1000000 if exp_type == "million" else 1000000000 if exp_type == "billion" else 1000000000000 if exp_type == "trillion" else 1;
            if (thousand_val.is_integer()):
                return int(thousand_val*exp_val),"{0} {1}".format(int(thousand_val),exp_type)
            else: #Float
                # Check if the last digit is 5 (mean a haft)
                frac,whole = math.modf(thousand_val); # Get dicimal
                if frac == 0.5:
                    # Decide to turn to a half or not
                    if (np.random.randint(0,2)==1):
                        return int(whole*100),"{0} {1} and a half".format(int(whole),exp_type)
                # Otherwise, just keep the original     
                return int(thousand_val*exp_val),"{0} {1}".format(thousand_val,exp_type)
        if (num<1000000000): #  from 0-999,999,999
            million_val = truncate_float(num/1000000,1);
            if (million_val.is_integer()):
                return int(million_val*1000000),"{0} million".format(int(million_val))
            else: #Float
                # Check if the last digit is 5 (mean a haft)
                frac,whole = math.modf(million_val); # Get dicimal
                if frac == 0.5:
                    # Decide to turn to a half or not
                    if (np.random.randint(0,2)==1):
                        return int(whole*100),"{0} million and a half".format(int(whole))
                # Otherwise, just keep the original     
                return int(million_val*1000000),"{0} million".format(million_val)
        if (num<1000000000000): #  from 0-999,999,999,999
            billion_val = truncate_float(num/1000000000,1);
            if (billion_val.is_integer()):
                return int(billion_val*1000000000),"{0} billion".format(int(billion_val))
            else: #Float
                # Check if the last digit is 5 (mean a haft)
                frac,whole = math.modf(billion_val); # Get dicimal
                if frac == 0.5:
                    # Decide to turn to a half or not
                    if (np.random.randint(0,2)==1):
                        return int(whole*100),"{0} billion and a half".format(int(whole))
                # Otherwise, just keep the original     
                return int(billion_val*1000000000),"{0} billion".format(billion_val)
        if (num<1000000000000000): #  from 0-999,999,999,999,999
            trillion_val = truncate_float(num/1000000000000,1);
            if (trillion_val.is_integer()):
                return int(trillion_val*1000000000000),"{0} trillion".format(int(trillion_val))
            else: #Float
                # Check if the last digit is 5 (mean a haft)
                frac,whole = math.modf(trillion_val); # Get dicimal
                if frac == 0.5:
                    # Decide to turn to a half or not
                    if (np.random.randint(0,2)==1):
                        return int(whole*100),"{0} trillion and a half".format(int(whole))
                # Otherwise, just keep the original     
                return int(trillion_val*1000000000000),"{0} trillion".format(trillion_val)
    # Then decide whether the origianl value should be turn to roman
    if np.random.choice(a=[True,False],p=[0.05,0.95]): # Return true, then Roman
         # Roman can't represent number 0. ¯(°_o)/¯ , and shouldn't greater than 5000
        while (num == 0 or num>4999):
            num = np.random.randint(1,4999)
        #print(num)
        return num,roman.toRoman(num).lower();
            
        
        
    return num,p.number_to_words(num);

##############################################################################

def truncate_float(f, n):
    '''Truncates/pads a float f to n decimal places without truncate_floating'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return float('.'.join([i, (d+'0'*n)[:n]]))

##############################################################################
    
def reduce_number(num):
        if (num<1000): # number from 0-99toolbar_value9
            return 100,"hundreds"

        if (num<1000000): # number from 0-999999
            if (np.random.randint(0,2)==1): # decide whether turn to "haft a"
                return 500,"haft a thousands"                
                
            return 1000,"thousands"
            
        if (num<1000000000): #  from 0-999,999,999
            if (np.random.randint(0,2)==1): # decide whether turn to "haft a"
                return 500000,"haft a millions"  
            return 1000000,"millions"
            
        if (num<1000000000000): #  from 0-999,999,999,999
            if (np.random.randint(0,2)==1): # decide whether turn to "haft a"
                return 500000000,"haft a billions" 
            return 1000000000,"billions"
           
        if (num<1000000000000000): #  from 0-999,999,999,999,999
            if (np.random.randint(0,2)==1): # decide whether turn to "haft a"
                return 500000000000,"haft a trillions" 
            return 1000000000000,"trillions"

##############################################################################

def generate_number(start_point,end_point,dist):
    #print(dist)
    if dist == "uniform":
        num = np.random.randint(start_point,end_point+1);
        rand_values.append(num);
        return num;
    elif dist == "gaussian": # Gausian
        num = np.random.random_integers(start_point,end_point);
        rand_values.append(num);
        #print(num)
        return num;
    
    # "exponential":
    num = -1; # Draft value 
    while num<start_point or num>end_point:
        num = int(np.random.exponential(scale=end_point/3))
        #if num>end_point:
        #    num = num%1000; # pull number to range 0-99
    #print(num)
    rand_values.append(num)       
    return num;

###############################################################################

def year_transformation(num,seq):
    """ Transform a year to its form +YYYYXXX (AD) -YYYXXXX. It took inputs are
    number (symbol sequence) and spelled out form (word form - input) """
    # decide before or after Christ
    chrs_type = np.random.choice(a=[0,1,2],p=[0.4,0.5,0.1]);
    if chrs_type==0:# Before Chrs
        word = seq + ("~b.c." if bool(random.getrandbits(1)) else "~bc");
        sym = "-"+str(num)+"XXXX"
    elif chrs_type==1:# Anno Domin, without AD
        word = seq;
        sym = "+"+str(num)+"XXXX"
    else:# Anno Domin, with AD (very rare)
        word = seq+ ("~a.d." if bool(random.getrandbits(1)) else "~ad");
        sym = "+"+str(num)+"XXXX"
    return sym,word
    
###############################################################################

def generate_number_seq(submode,withNoise = False, withReduce=False, err_indexs = [], size = 1000):
    # SUBMODE indidcate what kind of number you want it to return
    # 0: Regular
    # 1: Number + year
    # 2: #number and rank/ordinal
    # 3:  All the things above
    # WITHNOISE: the sample contains error or not?
    # WITHREDUCE: the sample will be reduced or not?
    if (np.random.sample()<0.2): # decide wether this number should be spell out
        # No spell out...
        num = generate_number(start_point,end_point,distribution_mode);
       #csv_file.write(addspaces(str(num),spc) + "," +addspaces(str(num),spc) + "\n");
        return addspaces(str(num),spc) , ("#QUC ~ " if spc else "#QUC~") + addspaces(str(num),spc);
        if i in err_indexs: # it was defined to have error, but in this case we exclude this sample
            while len(err_indexs) < int(size*ovr_err):  # Therefore we need to find and substitusion
                err_indexs.add(np.random.randint(i+1,size));
    elif (submode==0): # Regular number
        num = generate_number(start_point,end_point,distribution_mode);
        num,seq = convert_to_words(num);
        if withReduce:
            num,seq = reduce_number(num);
        if withNoise: # add error to this
            #csv_file.write(addspaces(str(num),spc) + "," +addspaces(addnoise(addtilde(seq),ovr_err),spc) + "\n");
            return addspaces(str(num),spc),("#QUC ~ " if spc else "#QUC~") + addspaces(addnoise(addtilde(seq),smp_err),spc);
        else:
            #csv_file.write(addspaces(str(num),spc) + "," +addspaces(addtilde(seq),spc) + "\n");
            return addspaces(str(num),spc),("#QUC ~ " if spc else "#QUC~") + addspaces(addtilde(seq),spc)
    elif (submode==1): # Number + year
        #randomly select a number type
        num_type =np.random.randint(0,2);
        sem_tag = ("#QUC ~ " if num_type == 0 else "#YOC ~ ") if spc else \
                  ("#QUC~" if num_type == 1 else "#YOC~");
        if (num_type==0): # Regular number
            num = generate_number(start_point,end_point,distribution_mode);
            num,seq = convert_to_words(num);
            if withReduce:
                num,seq = reduce_number(num);
        elif (num_type==1): # year
            num =np.random.randint(1101,10000); # from elevent and one to nity-nine nity-nine
            seq=p.number_to_words(int(num/100))+" "+p.number_to_words(num%100)     
            num,seq = year_transformation(num,seq)
        # decice noise and write to file
        if withNoise: # add error to this
            #csv_file.write(addspaces(str(num),spc) + "," +addspaces(addnoise(addtilde(seq),ovr_err),spc) + "\n");
            return addspaces(str(num),spc), sem_tag + addspaces(addnoise(addtilde(seq),smp_err),spc);
        else:
            seq = p.number_to_words(num);
            #csv_file.write(addspaces(str(num),spc) + "," +addspaces(addtilde(seq),spc) + "\n");
            return addspaces(str(num),spc),sem_tag + addspaces(addtilde(seq),spc);
    elif (submode==2): #number and rank/ordinal
        num_type =np.random.randint(0,2);
        sem_tag = ("#QUC ~ " if num_type == 0 else "#ORD ~ ") if spc else \
                  ("#QUC~" if num_type == 1 else "#ORD~");
        if (num_type==0): # Regular number
            num = generate_number(start_point,end_point,distribution_mode);
            num,seq = convert_to_words(num);
            if withReduce:
                num,seq = reduce_number(num);
        elif (num_type==1): # rank/ordinal
            num = generate_number(start_point,end_point,distribution_mode);
            seq = p.ordinal(num);
            #randomly select a ordinal type: 1st, or first?
            rank_type =np.random.randint(0,2);
            if (rank_type==0): # first
                seq=p.number_to_words(seq);
        # decice noise and write to file
        if withNoise: # add error to this
            #csv_file.write(addspaces(str(num),spc) + "," +addspaces(addnoise(addtilde(seq),ovr_err),spc) + "\n");
            return addspaces(str(num),spc),sem_tag + addspaces(addnoise(addtilde(seq),smp_err),spc);
        else:
             seq = p.number_to_words(num);
             #csv_file.write(addspaces(str(num),spc) + "," +addspaces(addtilde(seq),spc) + "\n");
             return addspaces(str(num),spc),sem_tag + addspaces(addtilde(seq),spc);
    elif (submode==3): # All the things
        num_type =0+3*np.random.random(); #(random number from 0 to 4)
        #print(num_type)
        sem_tag = "#ZZZ"; # Init semtag. I choose Z, because it has low dist in English lang
        if (num_type>=0 and num_type< 1.5): # Regular number
            num = generate_number(start_point,end_point,distribution_mode);
            #if num<999:
            #    print "Small value: "+str(num)
            num,seq = convert_to_words(num);
            sem_tag = ("#QUC ~ " if spc == 1 else "#QUC~");
            if withReduce:
                num,seq = reduce_number(num);
        elif (num_type< 2.1): # year & date
            num =np.random.randint(1101,10000); # from elevent and one to nity-nine nity-nine
            # Decide whether it is normal year or decades of Day of month/month of year
            date_type= np.random.random();
            if (date_type< 0.8): # 80% spelled out, since it is more complex
                # cases:
                #   0: spelled out #YOC. E.g: nineteen~ninety~one  =1991
                #   1: decade #DEC. E.g: 1920s
                #   2: Day of month #DOM
                seq=p.number_to_words(int(num/100))+" "+p.number_to_words(num%100)
                num,seq = year_transformation(num,seq)
                sem_tag = ("#YOC ~ " if spc == 1 else "#YOC~");
            elif (date_type< 0.9): # decade
                num = num - num %10;
                seq = str(num) + "s";
                num,seq = year_transformation(num,seq)
                sem_tag = ("#DEC ~ " if spc == 1 else "#DEC~");
            else: # Other stuffs: #DOM, #DOW, #MOY)
                date_type = np.random.choice(a=[0,1,2],p=[0.3,0.3,0.4]);                
                if date_type==0: # decide to be #DOM
                    num = np.random.randint(0,32); # get day number value 0-31
                    seq = p.ordinal(num) if np.random.randint(0,2) else str(num);
                    num = "+XXXXXX" + str(num);
                    sem_tag = ("#DOM ~ " if spc == 1 else "#DOM~");
                elif date_type==1: # #DOW
                    num,seq = random.choice(day_of_week);
                    sem_tag = ("#DOW ~ " if spc == 1 else "#DOW~");
                else: # #MOY
                    num,seq = random.choice(month_of_year);
                    sem_tag = ("#MOY ~ " if spc == 1 else "MOY~");
                        
            
        elif (num_type>=2.1 and num_type<= 2.9): # rank
            num = generate_number(start_point,end_point,distribution_mode);
            #if num<999:
            #    print "Small value: "+str(num)
            seq = p.ordinal(num);
            #randomly select a ordinal type: 1st, or first?
            rank_type =np.random.randint(0,2);
            if (rank_type==0): # first
                seq=p.number_to_words(seq);
            sem_tag = ("#ORD ~ " if spc == 1 else "ORD~");
        else:   #Score
            num1,num2 = np.random.randint(0,101,size=2);
            num = str(num1) + "-" + str(num2)
            seq = num;
            sem_tag = ("#SCO ~ " if spc == 1 else "SCO~");
##############################################################################f spc == 1 else "#ORD~");
        # decice noise and write to file
        if withNoise: # add error to this
            #csv_file.write(addspaces(str(num),spc) + "," +addspaces(addnoise(addtilde(seq),ovr_err),spc) + "\n");
            return addspaces(str(num),spc),(sem_tag + addspaces(addnoise(addtilde(seq),smp_err),spc));
        else:
            #seq = p.number_to_words(num);
            #csv_file.write(addspaces(str(num),spc) + "," +addspaces(addtilde(seq),spc) + "\n");
            return addspaces(str(num),spc),(sem_tag + addspaces(addtilde(seq),spc));

##############################################################################
def generate_time_seq(withNoise = False):
    # Chose a random number for time format:
    #   0-9: standard formats as in "formats"
    #    10: quarter~to~NUMBER
    #    11: quarter~to~WORD
    #    12: quarter~past~NUMBER
    #    13: quarter~past~WORD   
    #    14: half~to~NUMBER
    #    15: half~to~WORD
    #    16: half~past~NUMBER
    #    17: half~past~WORD
    #    18: midday,midnight, noon, noontide, noontime
    #    19: NUMBER~o'clock         -> WARNING Not yet fully suppoted
    #    20: WORD~o'clock           -> WARNING Not yet fully suppoted
    hours = np.random.randint(0,24)
    # Set probability of even cases (0,5,10,..,55) to be more common
    minutes_probabilities = [3/84 if i%5==0 else 1/84 for i in xrange(0,60)]
    #print minutes_probabilities
    minutes = np.random.choice(range(0,60), p=minutes_probabilities)
    tm = time(hour=hours,minute=minutes)
    #print tm.isoformat();
    time_format_index =  np.random.choice(range(21),p=time_format_probabilities); # choose a format
    rs=None;
    #print "1"
    if time_format_index in range(0,10): # 0-9: standard formats as in "formats"
        pattern = time_formats[time_format_index];
        rs = tm.strftime(pattern).lower();
        
        if time_format_index in range(2,10): # add dot randomly
            rs = re.sub("am$",np.random.choice(["am","am.","a.m","a.m."],p=[0.35,0.15,0.15,0.35]),rs)
            rs = re.sub("pm$",np.random.choice(["pm","pm.","p.m","p.m."],p=[0.35,0.15,0.15,0.35]),rs)   
        
        if time_format_index in range(6,10): # Eliminate minute for case 1pm
            tm = time(hour=tm.hour,minute=0) 
        
    if time_format_index in range(10,12): # quarter to 
        if tm.hour-1 >=0:
            tm = time(hour=(tm.hour-1),minute=45);
            rs = "quarter~to~"+ np.random.choice([tm.hour+1,addtilde(p.number_to_words(tm.hour+1))]);
        else:
            tm = time(hour=(tm.hour),minute=45);
            rs = "quarter~to~"+ np.random.choice([tm.hour+1,addtilde(p.number_to_words(tm.hour+1))]);
        
    if time_format_index in range(12,14): # quarter past 
        if tm.hour+1 <24:
            tm = time(hour=tm.hour,minute=15);
            rs = "quarter~past~"+ np.random.choice([tm.hour,addtilde(p.number_to_words(tm.hour))]);
        else:
            tm = time(hour=(tm.hour-1),minute=45);
            rs = "quarter~past~"+ np.random.choice([tm.hour-1,addtilde(p.number_to_words(tm.hour))]);
            
    if time_format_index in range(14,16): # half to 
        if tm.hour-1 >=0:
            tm = time(hour=(tm.hour-1),minute=30);
            rs = "half~to~"+ np.random.choice([tm.hour+1,addtilde(p.number_to_words(tm.hour+1))]);
        else:
            tm = time(hour=(tm.hour),minute=30);
            rs = "half~to~"+ np.random.choice([tm.hour+1,addtilde(p.number_to_words(tm.hour+1))]);
        
    if time_format_index in range(16,18): # half past 
        if tm.hour+1 <24:
            tm = time(hour=tm.hour,minute=30);
            rs = "half~past~"+ np.random.choice([tm.hour,addtilde(p.number_to_words(tm.hour))]);
        else:
            tm = time(hour=(tm.hour-1),minute=30);
            rs = "half~past~"+ np.random.choice([tm.hour-1,addtilde(p.number_to_words(tm.hour))]);
                
    if time_format_index == 18: # midday,midnight, noon, noontide, noontime
        dt=np.dtype('S10,S10')
        tmp =  np.random.choice(np.array(fixed_time_terms,dtype=dt),p=[0.35,0.13,0.35,0.07,0.1])
        if withNoise:
            return addspaces(tmp[0],spc),("#CLO ~ " if spc == 1 else "#CLO~") + addspaces(addnoise(tmp[1],smp_err),spc); 
        else:
            return addspaces(tmp[0],spc),("#CLO ~ " if spc == 1 else "#CLO~") + addspaces(tmp[1],spc); 
    
    # o'clock, only consider from 1-12 morning  INCOMPLETED    
    # Assumption: o'clock indicate morning, staring from 1 o'clock (01:00) to twelve o'clock (12:00)
    if time_format_index in range(19,21): 
        tm = time(hour=tm.hour,minute=0);
        rs = np.random.choice([tm.hour,addtilde(p.number_to_words(tm.hour))])+"~o'clock";
    
    #print "2"     
    if withNoise:
        return addspaces(tm.strftime("%H:%M"),spc),("#CLO ~ " if spc == 1 else "#CLO~") + addspaces(addnoise(rs,smp_err),spc); 
    
    return addspaces(tm.strftime("%H:%M"),spc), ("#CLO ~ " if spc == 1 else "#CLO~") + addspaces(rs,spc); 

###############################################################################

def generate_GPO_seq(withNoise = False):
    #sym,word = np.random.choice(np.chararray(gpo_list,unicode=True,dtype=np.dtype(('U',60),('U',30))));  
    sym,word = random.choice(gpo_list);
    if withNoise:
        return addspaces(sym.lower(),spc),(u"#GPO ~ " if spc == 1 else u"#GPO~") + addspaces(addnoise(word.lower(),smp_err),spc); 
    
    return addspaces(sym,spc), (u"#GPO ~ " if spc == 1 else u"#GPO~") + addspaces(word,spc); 
    
###############################################################################

def createEVElist():
    result = set([]);
    person_types = ["1","2","3","*"] 
    for verb in english_infinitive_verbs:
        # Check whether it is a verb:
        if (en.is_verb(verb)):
            infi = en.verb.infinitive(verb);
            # Group 1: SIMPLE
            # ::EXS untensed simple
            exs = infi;  # Untensed simple: be, walk, eat
            result.add((infi,exs,"#EXS"));
            # ::ENS (present simple) vary by subject types
            for person in person_types:
                ens = en.verb.present(infi,person=person);
                if ens:
                    result.add((infi,ens,"#ENS"));
            # ::EPS (past simple) 
            for person in person_types:
                eps = en.verb.past(infi,person=person);
                if eps:
                    result.add((infi,eps,"#EPS"));
            # !! ::EFS: no sample        
            # Group 2: PROGRESSIVE
            # ::EXG"
            exg = en.verb.present_participle(infi);  
            result.add((infi,exg,"#EXG")); # Untensed progressive: going, giving
            
            # ::ENG
            result.add((infi,exg,"#ENG")); # Present progressive: going, giving
            # ::EPG
            result.add((infi,exg,"#EPG")); # Past progressive: going, giving
            # ::EFG
            result.add((infi,exg,"#EFG")); # Future progressive: going, giving
            
            # Group 3: PERFECT
            # ::EXT
            ext = en.verb.past_participle(infi);  # Untensed perfect: gone, given
            result.add((infi,ext,"#EXT"));  
            
            # ::ENT
            result.add((infi,ext,"#ENT"));
            # ::EPT
            result.add((infi,ext,"#EPT"));  
            # ::EFT
            result.add((infi,ext,"#EFT"));
            ## Another expansion for EXS
            result.add((infi,ext,"#EXS"));
    #print("Created EVE list")                    
    return list(result)

###############################################################################

def generate_EVE_seq(withNoise = False,eventList=[]):
    if not eventList:
        print("Invalid event set!");
    sym,word,tag =  random.choice(eventList);
    if withNoise:
        return addspaces(sym.lower(),spc),((tag + " ~ ") if spc == 1 else (tag + "~")) + addspaces(addnoise(word.lower(),smp_err),spc); 
    return addspaces(sym,spc), ((tag + " ~ ") if spc == 1 else (tag + "~")) + addspaces(word,spc); 

###############################################################################

def generate_CON_seq(withNoise = False,conceptList=[]):
    if not conceptList:
        print("Invalid event set!");
    sym,word,tag =  random.choice(conceptList);
    if withNoise:
        return addspaces(sym.lower(),spc),((tag + " ~ ") if spc == 1 else (tag + "~")) + addspaces(addnoise(word.lower(),smp_err),spc); 
    return addspaces(sym,spc), ((tag + " ~ ") if spc == 1 else (tag + "~")) + addspaces(word,spc); 
    
###############################################################################

def generate_TAMD_seq(withNoise = False,tamdList=[]):
    if not tamdList:
        print("Invalid TMA set!");
    sym,word,tag =  random.choice(tamdList);
    if withNoise:
        return addspaces(sym.lower(),spc),((tag + " ~ ") if spc == 1 else (tag + "~")) + addspaces(addnoise(word.lower(),smp_err),spc); 
    return addspaces(sym,spc), ((tag + " ~ ") if spc == 1 else (tag + "~")) + addspaces(word,spc);
    
###############################################################################
    
def generate_ACDLT_seq(withNoise = False,acdList=[]):
    if not acdList:
        print("Invalid ANA - COM - DIS - LOG set!");
    sym,word,tag =  random.choice(acdList);
    if withNoise:
        return addspaces(sym.lower(),spc),((tag + " ~ ") if spc == 1 else (tag + "~")) + addspaces(addnoise(word.lower(),smp_err),spc); 
    return addspaces(sym,spc), ((tag + " ~ ") if spc == 1 else (tag + "~")) + addspaces(word,spc); 

###############################################################################

def generate_UOM_seq(withNoise = False,uomList=[]):
    if not uomList:
        print("Invalid UOM set!");
    sym,word,tag =  random.choice(uomList);
    if withNoise:
        return addspaces(sym.lower(),spc),((tag + " ~ ") if spc == 1 else (tag + "~")) + addspaces(addnoise(word.lower(),smp_err),spc); 
    return addspaces(sym,spc), ((tag + " ~ ") if spc == 1 else (tag + "~")) + addspaces(word,spc); 

###############################################################################

def make_reduced_indeces_set(choices= [], num_indicator=0, size = 1000):
    #print("Started making reduce set");
    #print("!!!!!!!" + str(len(choices)))
    #print(num_indicator)
    #rs = set([])
    # get list of numberal indeces:
    num_indeces =  [ i for i in range(len(choices)) if choices[i]==num_indicator];

    return random.sample(num_indeces,int(size*reduce_rate));

###############################################################################

def update_progress_bar(i=0,size=100):
    global toolbar_value;
    if (toolbar_value +1 == int(i*toolbar_width/size)): 
        toolbar_value =  int(i*toolbar_width/size);
        sys.stdout.write("█")
        sys.stdout.flush()
    if (i%1000==0):
        label = "%03.2f" % float(i/size)
        sys.stdout.write(" {0}".format(label));
        sys.stdout.flush()
        sys.stdout.write("\b" * (5)) # return to current progressin point
    
###############################################################################       

def generator(threadID,size=1000):
    #self.threadID= threadID;
    #self.size=size;
    if size<0:
        print("Incorect allocated job to thread "+threadID);
    else:
        out_file = "tmp/" + threadID + "tmp.csv"
        out_file_root = re.sub(".[a-z]+$","",out_file);
        csv_file =  codecs.open(out_file, encoding='utf-8', mode='w');
        num_file = codecs.open(out_file_root+".sym", encoding='utf-8', mode='w');
        word_file = codecs.open(out_file_root + ".word", encoding='utf-8', mode='w');
        # decide which records contains error
        err_indexs = random.sample(range(size), int(size*ovr_err))
        red_indexs = set([])
        if (mode in [0,1,2,3]): # mode 1,2,3
            # Decide samples that will be reduced
            while len(red_indexs) < int(size*reduce_rate): 
                red_indexs.add(np.random.randint(0,size));
            for i in xrange(size):
                # Generate sample
                wnoise = True if i in err_indexs else False;
                wreduce = True if i in red_indexs else False;
                rs = generate_number_seq(mode,withNoise = wnoise, withReduce=wreduce, err_indexs = err_indexs, size = size)
                #Write to file
                csv_file.write(u"{0},{1}\n".format(rs[0],rs[1]));
                # update progress bar
                #if (toolbar_value +1 == int(i*toolbar_width/size)):
                #   toolbar_value =  int(i*toolbar_width/size);
                #   sys.stdout.write("█")
                #   sys.stdout.flush()
                   
        elif (mode == 4): # mode 4
            for i in xrange(size):
                wnoise = True if i in err_indexs else False;
                rs = generate_time_seq(withNoise = wnoise)
                csv_file.write(u"{0},{1}\n".format(rs[0],rs[1]));
                # update progress bar
                #if (toolbar_value +1 == int(i*toolbar_width/size)):
                #   toolbar_value =  int(i*toolbar_width/size);
                #   sys.stdout.write("█")
                #   sys.stdout.flush()           
        elif (mode==5): # Number and time
            # make a array that decide wich sample to be number, which sample to be time
            choices = np.random.choice(a=[0,1],size=size,p=[0.5,0.5]); 
            #print(len(choices))
            # get list of numberal indeces:
            num_indeces =  [ i for i in range(len(choices)) if choices[i]==0];
            #print(len(num_indeces))
            
            # Decide samples that will be reduced
            while len(red_indexs) < int(size*reduce_rate):
                item =  np.random.choice(num_indeces)
                num_indeces.remove(item)
                red_indexs.add(item);
            #print("Number of number items: {0}.\n Number of reduced samples: {1}".format(len(num_indeces),len(red_indexs)))
            # Dispose the choices array
            #choices = None;
            i=0;
            for choice in choices:    
                if choice == 0: # number:
                    wnoise = True if i in err_indexs else False;
                    wreduce = True if i in red_indexs else False;
                    sym,word = generate_number_seq(3,withNoise = wnoise, withReduce=wreduce,err_indexs = err_indexs, size = size)
                    #print "--- " + sym + "    " + word
                    #Write to file
                    #csv_file.write(u"{0},{1}\n".format(sym,word));
                else: # choice==1: time
                    wnoise = True if i in err_indexs else False;
                    #rs = generate_time_seq(withNoise = wnoise)
                    sym,word = generate_time_seq(withNoise = wnoise)
                    #csv_file.write(u"{0},{1}\n".format(rs[0],rs[1]));
                csv_file.write(u"{0},{1}\n".format(sym,word));
                num_file.write(sym+"\n");
                word_file.write(word+"\n")
                i +=1;#update_progress_bar(i); # Update progress bar
                   
        elif (mode==6): # Number, Year, Time and GEO-political entities
            # make a array that decide wich sample to be number, which sample to be time, 
            choices = np.random.choice(a=[0,1,2],size=size,p=[0.50,0.47,0.03]);
            # get list of numberal indeces:
            num_indeces =  [ i for i in range(len(choices)) if choices[i]==0];
            while len(red_indexs) < int(size*reduce_rate):
                item =  np.random.choice(num_indeces)
                num_indeces.remove(item)
                red_indexs.add(item);    
            # Now do the selection
            i=0
            for choice in choices:
                if choice == 0: # number:
                    wnoise = True if i in err_indexs else False;
                    wreduce = True if i in red_indexs else False;
                    sym,word = generate_number_seq(3,withNoise = wnoise, withReduce=wreduce,err_indexs = err_indexs, size = size)
                    csv_file.write(u"{0},{1}\n".format(sym,word));
                elif choice==1: #time
                    wnoise = True if i in err_indexs else False;
                    #rs = generate_time_seq(withNoise = wnoise)
                    sym,word = generate_time_seq(withNoise = wnoise)
                    #csv_file.write(u"{0},{1}\n".format(rs[0],rs[1]));
                elif choice==2: #nationalities
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_GPO_seq(withNoise = wnoise)
                    #rs = u"{0},{1}\n".format(sym,word);
                    #csv_file.write(rs);
                csv_file.write(u"{0},{1}\n".format(sym,word));
                num_file.write(sym+"\n"); word_file.write(word+"\n")
                # Update status bar
                i +=1;#update_progress_bar(i); # Update progress bar
                    
        elif (mode==7): # Number, Year, Time and GEO-political entities and  nts (ﾉﾟ0ﾟ)ﾉ
            # Create event set, from all verbs
            # each element is: (tar_w,src_w,sem) where: 
            #- tar_w is target word (e.g "be")
            #- src_w is source word (e.g "was")
            #- src_sem: the the semantic tag of the source word
            #eveList = createEVElist();
            # Assign data type rows     
            choices = np.random.choice(a=[0,1,2,3],size=size,p=[0.4,0.35,0.03,0.22]);
            # 0: number, 1: time, 2: nationalities, 3: EVE
            # get list of numberal indeces:
            num_indeces =  [ i for i in range(len(choices)) if choices[i]==0];
            while len(red_indexs) < int(size*reduce_rate):
                item =  np.random.choice(num_indeces)
                num_indeces.remove(item)
                red_indexs.add(item);
            # Now do the selection
            i=0
            for choice in choices:
                if choice == 0: # number:
                    wnoise = True if i in err_indexs else False;
                    wreduce = True if i in red_indexs else False;
                    sym,word = generate_number_seq(3,withNoise = wnoise, withReduce=wreduce,err_indexs = err_indexs, size = size)
                    #csv_file.write(u"{0},{1}\n".format(sym,word));
                elif choice==1: #time
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_time_seq(withNoise = wnoise)
                    #csv_file.write(u"{0},{1}\n".format(rs[0],rs[1]));
                elif choice==2: #nationalities
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_GPO_seq(withNoise = wnoise)
                    #rs = u"{0},{1}\n".format(nation,national);
                    #csv_file.write(rs);
                elif choice==3: # EVE tags
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_EVE_seq(withNoise = wnoise, eventList=eveList)
                    #csv_file.write(u"{0},{1}\n".format(sym,word));
                csv_file.write(u"{0},{1}\n".format(sym,word));
                num_file.write(sym+"\n"); word_file.write(word+"\n");
                i +=1;#update_progress_bar(i); # Update progress bar
              
        elif (mode==8): # Number, Year, Time and GEO-political entities and Events, TNS and MOD and ACT (◕_◕)
            # First join the TNS,MOD and ACT together
            tns_mod_act_dem = [];
            tns_mod_act_dem.extend(tns_list);tns_mod_act_dem.extend(act_list);
            tns_mod_act_dem.extend(mod_list);tns_mod_act_dem.extend(dem_list);
            #eveList = createEVElist();
            choices = np.random.choice(a=[0,1,2,3,4],size=size,p=[0.39,0.35,0.03,0.2,0.03]);
            red_indexs = make_reduced_indeces_set(choices = choices);
            i=0
            for choice in choices:
                if choice == 0: # number:
                    wnoise = True if i in err_indexs else False;
                    wreduce = True if i in red_indexs else False;
                    sym,word = generate_number_seq(3,withNoise = wnoise, withReduce=wreduce,err_indexs = err_indexs, size = size)
                elif choice==1: #time
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_time_seq(withNoise = wnoise)
                elif choice==2: #nationalities
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_GPO_seq(withNoise = wnoise)
                elif choice==3: # EVE tags
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_EVE_seq(withNoise = wnoise, eventList=eveList)
                elif choice==4: # TNS & MOD & ACT & DEM tags
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_TAMD_seq(withNoise = wnoise, tamdList=tns_mod_act_dem)
                csv_file.write(u"{0},{1}\n".format(sym,word));
                num_file.write(sym+"\n"); word_file.write(word+"\n");
                i+=1; #update_progress_bar(i); # Update progress bar
                
        elif (mode==9): # Number, Year, Time and GEO-political entities and Events, TNS, MOD, ACT, DEM, ANA, COM, DIS and LOG(◕_◕)
            # First join the TNS,MOD and ACT together
            tns_mod_act_dem = [];
            tns_mod_act_dem.extend(tns_list);tns_mod_act_dem.extend(act_list);
            tns_mod_act_dem.extend(mod_list);tns_mod_act_dem.extend(dem_list);
            ana_com_dis_log_ttl = [];ana_com_dis_log_ttl.extend(dis_list)
            ana_com_dis_log_ttl.extend(com_list);ana_com_dis_log_ttl.extend(ana_list);
            ana_com_dis_log_ttl.extend(log_list);
            #print(len(ana_com_dis_log_ttl))
            #eveList = createEVElist();
            choices = np.random.choice(a=[0,1,2,3,4,5],size=size,p=[0.31,0.22,0.03,0.23,0.03,0.18]);
            # 0: number, 1: time, 2: nationalities, 3: EVE, 4. TNS & MOD & ACT 5. ANA & COM
            red_indexs = make_reduced_indeces_set(choices = choices);
            i=0
            for choice in choices:
                if choice == 0: # number:
                    wnoise = True if i in err_indexs else False;
                    wreduce = True if i in red_indexs else False;
                    sym,word = generate_number_seq(3,withNoise = wnoise, withReduce=wreduce,err_indexs = err_indexs, size = size)
                elif choice==1: #time
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_time_seq(withNoise = wnoise)
                elif choice==2: #nationalities
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_GPO_seq(withNoise = wnoise)
                elif choice==3: # EVE tags
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_EVE_seq(withNoise = wnoise, eventList=eveList)
                elif choice==4: # TNS & MOD & ACT & DEM tags
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_TAMD_seq(withNoise = wnoise, tamdList=tns_mod_act_dem)
                elif choice==5: # ANA & COM & DIS
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_ACDLT_seq(withNoise = wnoise, acdList=ana_com_dis_log_ttl)
                csv_file.write(u"{0},{1}\n".format(sym,word));
                num_file.write(sym+"\n"); word_file.write(word+"\n");
                i+=1; #update_progress_bar(i); # Update progress bar
        
        elif (mode==10): # Number, Year, Time and GEO-political entities and Events, TNS, MOD, ACT, DEM, ANA, COM, DIS, LOG, CON(◕_◕)
            # First join the TNS,MOD and ACT together
            tns_mod_act_dem = [];
            tns_mod_act_dem.extend(tns_list);tns_mod_act_dem.extend(act_list);
            tns_mod_act_dem.extend(mod_list);tns_mod_act_dem.extend(dem_list);
            ana_com_dis_log_ttl = [];ana_com_dis_log_ttl.extend(dis_list)
            ana_com_dis_log_ttl.extend(com_list);ana_com_dis_log_ttl.extend(ana_list);
            ana_com_dis_log_ttl.extend(log_list);ana_com_dis_log_ttl.extend(ttl_list)
            #print(len(ana_com_dis_log_ttl))
            #eveList = createEVElist();
            #wordnet_nouns = en.wordnet.wordnet.N;
            #conList = build_concept_list(wordnet_nouns);
            choices = np.random.choice(a=[0,1,2,3,4,5,6],size=size,p=[0.24,0.17,0.03,0.18,0.03,0.13,0.22]);
            # 0: number, 1: time, 2: nationalities, 3: EVE, 4. TNS & MOD & ACT 5. ANA & COM, 6.CON
            red_indexs = make_reduced_indeces_set(choices = choices);
            i=0
            for choice in choices:
                if choice == 0: # number:
                    wnoise = True if i in err_indexs else False;
                    wreduce = True if i in red_indexs else False;
                    sym,word = generate_number_seq(3,withNoise = wnoise, withReduce=wreduce,err_indexs = err_indexs, size = size)
                elif choice==1: #time
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_time_seq(withNoise = wnoise)
                elif choice==2: #nationalities
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_GPO_seq(withNoise = wnoise)
                elif choice==3: # EVE tags
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_EVE_seq(withNoise = wnoise, eventList=eveList)
                elif choice==4: # TNS & MOD & ACT & DEM tags
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_TAMD_seq(withNoise = wnoise, tamdList=tns_mod_act_dem)
                elif choice==5: # ANA & COM & DIS & LOGICAL & TILE
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_ACDLT_seq(withNoise = wnoise, acdList=ana_com_dis_log_ttl)
                elif choice==6: # CON ◘_◘ this could take a looooooong time...
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_CON_seq(withNoise = wnoise, conceptList=conList)
                csv_file.write(u"{0},{1}\n".format(sym,word));
                num_file.write(sym+"\n"); word_file.write(word+"\n");
                i+=1; #update_progress_bar(i); # Update progress bar
                if i%1000==0:
                    print("Thread {0} generated {1} samples out of {2}".format(threadID,i,size))
            
        elif (mode==11): # Number, Year, Time and GEO-political entities and Events, TNS, MOD, ACT, DEM, ANA, COM, DIS, LOG, CON, UOM and TTL(◕_◕)
            # First join the TNS,MOD and ACT together
            tns_mod_act_dem = [];
            tns_mod_act_dem.extend(tns_list);tns_mod_act_dem.extend(act_list);
            tns_mod_act_dem.extend(mod_list);tns_mod_act_dem.extend(dem_list);
            ana_com_dis_log_ttl = [];
            ana_com_dis_log_ttl.extend(dis_list)
            ana_com_dis_log_ttl.extend(com_list);ana_com_dis_log_ttl.extend(ana_list);
            ana_com_dis_log_ttl.extend(log_list);ana_com_dis_log_ttl.extend(ttl_list)
            #print(len(ana_com_dis_log_ttl))
            #eveList = createEVElist();
            #print("[i] Thread {0} created EVE list".format(threadID))
            #wordnet_nouns = en.wordnet.wordnet.N;
            #conList = build_concept_list(wordnet_nouns);
            #print("[i] Thread {0} created CON list".format(threadID))
            choices = np.random.choice(a=[0,1,2,3,4,5,6,7],size=size,p=[0.29,0.13,0.02,0.11,0.03,0.11,0.18,0.13]);
            # 0: number, 1: time, 2: nationalities, 3: EVE, 4. TNS & MOD & ACT 5. ANA & COM 6.CON, 7.UOM
            red_indexs = make_reduced_indeces_set(choices = choices);
            i=0
            print("[i] Thread {0} started mass generation.".format(threadID))
            for choice in choices:
                if choice == 0: # number:
                    wnoise = True if i in err_indexs else False;
                    wreduce = True if i in red_indexs else False;
                    sym,word = generate_number_seq(3,withNoise = wnoise, withReduce=wreduce,err_indexs = err_indexs, size = size)
                elif choice==1: #time
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_time_seq(withNoise = wnoise)
                elif choice==2: #nationalities
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_GPO_seq(withNoise = wnoise)
                elif choice==3: # EVE tags
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_EVE_seq(withNoise = wnoise, eventList=eveList)
                elif choice==4: # TNS & MOD & ACT & DEM tags
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_TAMD_seq(withNoise = wnoise, tamdList=tns_mod_act_dem)
                elif choice==5: # ANA & COM & DIS
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_ACDLT_seq(withNoise = wnoise, acdList=ana_com_dis_log_ttl)
                elif choice==6: # CON ◘_◘ this could take a looooooong time...
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_CON_seq(withNoise = wnoise, conceptList=conList)
                elif choice==7: # UOM O_O this could take a looooooong time...
                    wnoise = True if i in err_indexs else False;
                    sym,word = generate_UOM_seq(withNoise = wnoise, uomList=uom_list)
                csv_file.write(u"{0},{1}\n".format(sym,word));
                num_file.write(sym+"\n"); word_file.write(word+"\n");
                i+=1; #update_progress_bar(i); # Update progress bar
                if i%10000==0:
                    print("Thread {0} generated {1} samples out of {2}".format(threadID,i,size))
                    
    csv_file.close();
    num_file.close();
    word_file.close();
###############################################################################   

if __name__ =='__main__':
    # initialize the inflict
    p = inflect.engine();
    try: # parse value from input
     #   python text2num_dataset_generate_c2c.py 3 50000 0.1 0.1 1 out.csv 0 999999 exponential
     #                         sys.argv[0]       1   2    3   4  5    6    7    8       9  
        print(sys.argv)        
        mode = int(sys.argv[1]);
        dataset_size = int(sys.argv[2]);
        ovr_err = float(sys.argv[3]);
        smp_err = float(sys.argv[4]);
        reduce_rate = math.pow(ovr_err*smp_err,1.5)
        print reduce_rate;
        spc = int(sys.argv[5]);
        out_file = sys.argv[6];
        if mode in [0,1,2,3,5,6,7]: 
            start_point = int(sys.argv[7]);
            end_point = int(sys.argv[8]);
            #print("!!!!!!!!!!" + str(end_point));
            assert (end_point >= start_point), "end point smaller than start point?"
            if len(sys.argv)>=10:
                dmode = sys.argv[9]; 
                if dmode in ["exponential","uniform","gaussian"]:
                    distribution_mode = dmode;
        
        
        
        print("- start_point: {0} ({1})\n- end_point: {2}({3})\n- Distribution mode: {4}".format(
                                start_point,
                                p.number_to_words(start_point),
                                end_point,
                                p.number_to_words(end_point),
                                distribution_mode
                                ))
    except ValueError:
        print("ERROR parsing input params")
        sys.exit();
           
    # Decide samples that will contain error
    #print("Generating err sample index list");
    #err_indexs = set([]);
    #index_ls = range(0,size)
    #while len(err_indexs) < int(size*ovr_err):
    #    old_len = len(err_indexs);
    #    ran_val = np.random.choice(index_ls);
    #    err_indexs.add(ran_val);
    #    if len(err_indexs)>old_len: # new item added
    #        index_ls.remove(ran_val);
    #err_indexs = random.sample(range(size), int(size*ovr_err))
    
    #print("Generating err sample index list-> Done");
    # A set to store items to be reduced to hundreds, thousand, millions...
    #red_indexs = set([])
    
    # setup toolbar
    #sys.stdout.write("╣%s╠" % (" " * toolbar_width))
    #sys.stdout.flush()
    #sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    
    # First generate the thread ID
    
    batch_size = int(dataset_size/NUMBER_OF_GENERATOR);
    odd = dataset_size%NUMBER_OF_GENERATOR;
    
    theadIds =[]
    for i in xrange(NUMBER_OF_GENERATOR):
        if (i<NUMBER_OF_GENERATOR-1):
           theadIds.append((str(i),batch_size));
        else:
            theadIds.append((str(i),batch_size+odd))
            
    print("[i] Triggering threads...")
    threads = []
    
    if mode>=7:
        eveList = createEVElist();
        print("[i] Built event list.")
    
    if mode>=10:
        conList = build_concept_list(en.wordnet.wordnet.N);
        print("[i] Built concept list.")
            
    for tinfo in theadIds:
        t = threading.Thread(target=generator,args=(tinfo[0],tinfo[1]))
        threads.append(t)
        t.start()
        print("[i] Thread {0} started!".format(tinfo[0]))
           
    
    #sys.stdout.write("\n\n")
    
    # Draw histogram
    #if rand_values: # There are numeral values inside
    #    print("Number of numeral records: " + str(len(rand_values)))
    #    rand_values=np.asarray(rand_values);
    #    plt.hist(rand_values, bins=100)  # plt.hist passes it's arguments to np.histogram
    #    #plt.show()
    #    
    for thread in threads:
        thread.join();
        
    print("Program completed")
    
    
