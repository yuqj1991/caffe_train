#!/usr/bin/env python

import numpy as np

def loadCSVFile(file_name):
    file_content = np.loadtxt(file_name, dtype=np.str, delimiter=",")
    return file_content

