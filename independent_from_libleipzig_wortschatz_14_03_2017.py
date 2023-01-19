# -*- coding: utf-8 -*-

import pandas as pd
from libleipzig import *



def get_synsets(txtfile):
    lob = open(txtfile, "r")
    readin = lob.read().split(',')

    outp = []
    for s in readin:
        #remove * from each string
        outp = [x.strip() for x in readin]
        #split each element (word + topics)
    del outp[-1]
    return outp
    	




