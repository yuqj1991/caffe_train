# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

import os
import codecs

class sampledata():

    global _sample_person
    global _sample_negative
    global _sample
    global _sample_label
    

    def __init__(self):
        self._sample_person = {}
        self._sample_negative = {}
        self._sample = []
        self._sample_label = {}
        lines = open('../data/train_val.txt','r')
        for line in lines:
            personname = line.split('@')[0]
            picname = line.split(' ')[0]
            self._sample.append(picname)
            if personname in self._sample_person.keys():
                self._sample_person[personname].append(picname)
            else:
                self._sample_person[personname] = []
                self._sample_person[personname].append(picname)
            self._sample_label[personname] = int(line.split(' ')[1])
        print len(self._sample_person)

if __name__ == '__main__':

    sample = sampledata()
    #print sample._sample
