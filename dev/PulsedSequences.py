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
# import dev.systemFunctions as systemFunctions
from IPython.display import clear_output
from tqdm import tqdm
import sys
import mdt69x
import time
import keyboard



def plot_this_xy(xdata, ydata, ax, title, xlabel, ylabel, linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2):
    ax.set_title(title, fontsize=20)
    ax.plot(xdata, ydata, linestyle = linestyle, color = color, linewidth = linewidth, marker = marker, markersize = markersize)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

class Rabi:
    def __init__(self, config, ps=None, sg386=None, rm=None):
        self.config = config
        self.ps = ps
        self.sg386 = sg386
        self.rm = rm
        
        # Validate that required instruments are provided
        if ps is None or sg386 is None:
            print("Warning: Some instruments not provided. Use set_instruments() to set them later.")
    
    def set_instruments(self, ps, sg386, rm=None):
        """Set instrument handles after initialization"""
        self.ps = ps
        self.sg386 = sg386
        self.rm = rm
    
    def validate_instruments(self):
        """Check if all required instruments are available"""
        if self.ps is None:
            raise RuntimeError("PulseStreamer not set. Call set_instruments() or pass ps to __init__")
        if self.sg386 is None:
            raise RuntimeError("SG386 not set. Call set_instruments() or pass sg386 to __init__")
    
    def create_pulse_sequence(self, aomvolt=None):
        """Create pulse sequence for ODMR measurement"""
        self.validate_instruments()
        
        aomvolt = aomvolt or self.config['aomvolt']
        
        # Create pulse sequence
        ch0patt = [(96, 1)]  # Laser always ON
        ch1patt = [(self.config['wait_t'], 1), 
                   (self.config['count_t'], 1), 
                   (self.config['addl_t'], 1),
                   (self.config['wait_t'], 0), 
                   (self.config['count_t'], 0), 
                   (self.config['addl_t'], 0),
                   (self.config['separation_t'], 0)]
        ch2patt = [(self.config['wait_t'], 0), 
                   (self.config['count_t'], 1), 
                   (self.config['addl_t'], 0),
                   (self.config['wait_t'], 0), 
                   (self.config['count_t'], 1), 
                   (self.config['addl_t'], 0),
                   (self.config['separation_t'], 0)]
        
        seq = self.ps.createSequence()
        seq.setDigital(0, ch0patt)
        seq.setDigital(1, ch1patt)
        seq.setDigital(2, ch2patt)
        
        return seq