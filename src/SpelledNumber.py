# -*- coding: utf-8 -*-
""" 
@project: SEN text symbolizer - Parallel meaning bank - RUG

@author: duynd

Define class SPELLEDNUMBER to store a spelled number in words

Beside funtion preprocess the number. For example, the string "9.9 billions", 
after sequenceFetch will come up an object has
 obj.texts.billion = "9.9"              # All the other texts.* = ""
 obj.values.billion= 0.0 (default)      # All the other values.* = 0.0
 obj.flags.billion = False (default)    # All the other flags.* = False

During this pre-processing, we turn "9.9" (obj.texts.billion) to 9.9 (float in 
obj.values.billion) and  obj.flags.billion = True.

Other flags will also be turn into True, because their texts. hold empty string

Once all the sequences of text are fully extracted, the initializer also compu-
te the actual value.
"""
import math;

class SpelledNumber(object):
    def __init__(self, trillion="", billion="", million="", thousand="", 
                 unit="zero"):
        self._texts = textSeq(trillion,billion,million,thousand,unit);
        self._values = valueSeq();
        self._flags = flagSeq();
        self._actualValue = None;
        # Some innitialization
        if (trillion==""): self._flags.trillion=True;
        else:
            try:
                self._values.trillion = float(self._texts.trillion)
                self._flags.trillion = True
            except:
                pass;
                
        if (billion==""): self._flags.billion=True;
        else:
            try:
                self._values.billion = float(self._texts.billion)
                self._flags.billion = True
            except:
                pass;
                
        if (million==""): self._flags.million=True;
        else:
            try:
                self._values.million = float(self._texts.million)
                self._flags.million = True
            except:
                pass;
                
        if (thousand==""): self._flags.thousand=True;
        else:
            try:
                self._values.thousand = float(self._texts.thousand)
                self._flags.thousand = True
            except:
                pass;
                
        if (unit==""): self._flags.unit=True;
        else:
            try:
                self._values.unit = float(self._texts.unit)
                self._flags.unit = True
            except:
                pass;
        
        #compute actual value
        if (self._flags.trillion and self._flags.billion and \
            self._flags.million and self._flags.thousand and self._flags.unit): 
            self._actualValue = self._values.trillion * math.pow(10,12) + \
            self._values.billion * math.pow(10,9) + \
            self._values.million * math.pow(10,6) + \
            self._values.thousand * math.pow(10,3) + self._values.unit; 
        
    @property
    def texts(self):
        return self._texts

    @property
    def values(self):
        return self._value

    @property
    def flags(self):
        return self._flags
    
    @property
    def actualValue(self):
        return self._actualValue
           
    def toString(self):       
        return "{ texts:[ tr: " + self._texts.trillion + " , " + \
                         "bi: "+ self._texts.billion  + " , " + \
                         "mi: "+ self._texts.million  + " , " + \
                         "th: "+ self._texts.thousand  + " , " + \
                         "un: "+ self._texts.unit +  "], " + \
                " values:[ tr: " + str(self._values.trillion) + " , " + \
                          "bi: " + str(self._values.billion)  + " , " + \
                          "mi: " + str(self._values.million)  + " , " + \
                          "th: " + str(self._values.thousand) + " , " + \
                          "un: " + str(self._values.unit) + "] "+ \
                " flags:[ tr: "  + str(self._flags.trillion)  + " , " + \
                          "bi: " + str(self._flags.billion)  + " , " + \
                          "mi: " + str(self._flags.million)  + " , " + \
                          "th: " + str(self._flags.thousand)  + " , " + \
                          "un: " + str(self._flags.unit) + "]}";
        #print(res);
        #return res;

 
###############################################################################

# Class to store the TEXT string
class textSeq(object):  
    def __init__(self, trillion="", billion="", million="", thousand="", 
                 unit="zero"):
        self.trillion= trillion;
        self.billion= billion;
        self.million= million;
        self.thousand= thousand;
        self.unit = unit;

# Class to store the VALUE of string (float)
class valueSeq(object):
    def __init__(self, trillion=0.0, billion=0.0, million=0.0, thousand=0.0, 
                 unit=0.0):
        self.trillion= trillion;
        self.billion= billion;
        self.million= million;
        self.thousand= thousand;
        self.unit = unit;
# Class to store the FLAGs. Flag is True mean the corresponding level was 
# correactly transform from text -> num
class flagSeq(object):
    def __init__(self, trillion=False, billion=False, million=False, 
                 thousand=False, unit=False):
        self.trillion= trillion;
        self.billion= billion;
        self.million= million;
        self.thousand= thousand;
        self.unit = unit;