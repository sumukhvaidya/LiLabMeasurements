import numpy as np
import time
import scipy
import matplotlib
import matplotlib.pyplot as plt
import pulsestreamer
import nifpga
import pyvisa
import serial
import io
import dev.systemFunctions as systemFunctions
from IPython.display import clear_output
from tqdm import tqdm
import sys
import mdt69x
import time
import keyboard

config={'aomvolt':0.5,
        'pulsenum':10000,
        'count_t':.05*1e6,
        'separation_t':.03*1e6,
        'addl_t':.003*1e6,
        'wait_t':.003*1e6,
        'freq':1.85e9,
        'mw_power':-9,
        'seqplot':True}

class ODMR:
    def __init__(self, config):
        self.config = config    
        return
    
    def get_contrast(self):

        return
    