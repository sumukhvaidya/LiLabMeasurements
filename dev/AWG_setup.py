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
# import mdt69x
import time
import keyboard
from zhinst.toolkit import Session,  Waveforms
from zhinst.toolkit import CommandTable

LiLabColormap =matplotlib.colors.ListedColormap(np.loadtxt('C:\\Users\\LiLabDesktop\\Desktop\\Sumukh\\LiLabMeasurements\\LiLabMeasurements\\Main\\colormap_LiLab.csv', delimiter=','), name='Lilab', N=None)


def plot_this_xy(xdata, ydata, ax, title, xlabel, ylabel, linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2):
    ax.set_title(title, fontsize=20)
    ax.plot(xdata, ydata, linestyle = linestyle, color = color, linewidth = linewidth, marker = marker, markersize = markersize)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)


class AWGSetup:
    def __init__(self, config, awg, awg2 = None):
        self.config = config
        self.awg = awg
        self.awg2 = awg2

    def Rabi_2_channel_setup(self):
        """Run Rabi measurement over a tau range"""
        ############ AWG setup ############
        # sequence parameters
        nu = self.config['nu_IQ'] #385 #385 #100 #100#388#385.7 # microwave frequency in MHz
        Fs = 2400 
        NUM_SUBCYCLES = self.config['n_points']  # Number of subcycles #Number of points
        tau_step0 =self.config['tau_step'] # tau step in us
        tau_start0 = self.config['tau_start'] # starting tau in us
        # t_tot = 78 # total time of a pulse frame in us
        t_tot = self.config['init_t'] + self.config['readout_t'] + self.config['wait_t'] + self.config['delay_t'] # total time of a pulse frame in us
        TT_tot = t_tot*NUM_SUBCYCLES*2  # total time need for measuring all data points once
        t_initi = self.config['init_t'] #laser initialization time
        A = self.config['IQ_amplitude'] # 0.65#0.5087 #0.6 #0.3393#0.3#0.7
        # A = 0.36 # 100 T=35ns
        tau_range = np.linspace(tau_start0, tau_start0 + tau_step0 * (NUM_SUBCYCLES - 1), NUM_SUBCYCLES)  # tau range in us 

        # convert parameters
        TAU_STEP = int(tau_step0*Fs) # convert tau step into tick
        #TAU_START = int(tau_start0*Fs) # convert tau step into tick
        DATA_POINTS_PER_SUBCYCLE = int(t_tot*Fs)  # Data points per subcycle
        DATA_POINTS_TOTAL = DATA_POINTS_PER_SUBCYCLE*NUM_SUBCYCLES*2 # total waveform points for measuring all data points once
        INITIAL_PERIOD_LENGTH =  int(tau_start0*Fs)  # Initial tau
        MAX_PERIOD_LENGTH = INITIAL_PERIOD_LENGTH + (NUM_SUBCYCLES - 1) * TAU_STEP  # Maximum length of tau
        WFM_LEN = NUM_SUBCYCLES * DATA_POINTS_PER_SUBCYCLE*2  # Total waveform length
        T_INITIAL = int((t_initi+0.12)*Fs)


        # define sequence veriable and send it the sequencer in LabOne

        seqc_a = """\
        const LENGTH = """
        seqc_b = str(DATA_POINTS_TOTAL)
        seqc_c = """; // Number of subcycles * Data points per subcycle

        wave w1 = placeholder(LENGTH, false, false); // Create a waveform of size LENGTH, with one marker
        wave w2 = placeholder(LENGTH, false, false); // Create a waveform of size LENGTH, with one marker

        assignWaveIndex(w1,w2,10);                 // Create a wave table entry with placeholder waveform              
                                                // routed to output 1, with index 10

        // execute the command table entry
        executeTableEntry(0);

        while(true){
        waitDigTrigger(1);
        playWave(w1,w2);

        waitWave();
        };
        """

        seqc_program = seqc_a + seqc_b +seqc_c
        # seqc_program = seqc_b
        print(seqc_program)

        
        WAVE_INDEX = 0
        TABLE_INDEX = 0

        # load the sequence
        self.awg.load_sequencer_program(seqc_program)

        wave1 = np.zeros(WFM_LEN)
        wave2 = np.zeros(WFM_LEN)

        # Generate sine waves for each subcycle
        for subcycle in range(NUM_SUBCYCLES):
            current_length = INITIAL_PERIOD_LENGTH + subcycle * TAU_STEP  # Length for the current sine wave
            start_index = subcycle * DATA_POINTS_PER_SUBCYCLE*2 + T_INITIAL # Center position
            
            # Generate sine wave
            for i in range(current_length):
                wave1[start_index + i] = A*np.sin( 2*np.pi*i*nu/Fs)  # Sine wave calculation

        # Generate cosine waves for each subcycle
        for subcycle in range(NUM_SUBCYCLES):
            current_length = INITIAL_PERIOD_LENGTH + subcycle * TAU_STEP  # Length for the current sine wave
            start_index = subcycle * DATA_POINTS_PER_SUBCYCLE*2 + T_INITIAL# Center position
            
            # Generate sine wave
            for i in range(current_length):
                wave2[start_index + i] = A*np.cos( 2*np.pi*i*nu/Fs)  # Sine wave calculation

        ## Upload waveforms

        waveforms = Waveforms()
        waveforms[10] = (wave1,wave2,None) # I-component (output channel 1), Q-component (output channel 2), marker
        # awg.commandtable.upload_to_device(ct)
        self.awg.write_to_waveform_memory(waveforms)
        self.awg.enable_sequencer(single=True)

    def CW_ODMR_2Channel(self):
        seqc_program = """\

        const LENGTH = 240000;

        wave w1 = placeholder(LENGTH, false, false); // Create a waveform of size LENGTH, with one marker
        wave w2 = placeholder(LENGTH, false, false); // Create a waveform of size LENGTH, with one marker

        assignWaveIndex(w1,w2,10);                 // Create a wave table entry with placeholder waveform              
                                                // routed to output 1, with index 10

        // execute the command table entry
        executeTableEntry(0);

        while(true){
        waitDigTrigger(1);
        playWave(w1,w2);

        waitWave();
        };
        """
        AWG_INDEX = 0 #use channel 1&2 for 2*2
        # awg = device.awgs[AWG_INDEX]
        WAVE_INDEX = 0
        TABLE_INDEX = 0

        # load the sequence
        self.awg.load_sequencer_program(seqc_program)


        # initialize the command table
        ct_schema = self.awg.commandtable.load_validation_schema()
        ct = CommandTable(ct_schema)

        # Wavefrom eith amplitude and phase settings
        ct.table[TABLE_INDEX].waveform.index = WAVE_INDEX
        ct.table[TABLE_INDEX].amplitude0.value = 1
        ct.table[TABLE_INDEX].amplitude1.value = 1


        ##Generate a waveform and marker
        Fs = 2400 # in MHz
        nu = self.config['nu_IQ'] # microwave frequency in MHz
        t_mw = int(self.config['count_t'] / 1000) # MW duration convterted to us
        A = self.config['IQ_amplitude'] # amplitude

        LENGTH = t_mw * Fs
        # number of points
        wave1 = A*np.sin(np.linspace(0, 2*np.pi*nu*t_mw, LENGTH))
        wave2 = A*np.cos(np.linspace(0, 2*np.pi*nu*t_mw, LENGTH))
        # marker = np.concatenate([np.ones(32), np.zeros(LENGTH-32)]).astype(int)

        ## Upload waveforms

        print(seqc_program)

        waveforms = Waveforms()
        waveforms[10] = (wave1,wave2,None) # I-component (output channel 1), Q-component (output channel 2), marker
        self.awg.commandtable.upload_to_device(ct)
        self.awg.write_to_waveform_memory(waveforms)
        self.awg.enable_sequencer(single=True)
        # awg.wait_done()

    def CW_ODMR_4Channel(self):
        seqc_program = """\

        const LENGTH = 240000;

        wave w1 = placeholder(LENGTH, false, false); // Create a waveform of size LENGTH, with one marker
        wave w2 = placeholder(LENGTH, false, false); // Create a waveform of size LENGTH, with one marker

        assignWaveIndex(w1,w2,10);                 // Create a wave table entry with placeholder waveform              
                                                // routed to output 1, with index 10

        // execute the command table entry
        executeTableEntry(0);

        while(true){
        waitDigTrigger(1);
        playWave(w1,w2);

        waitWave();
        };
        """

        seqc_program2 = """\

        const LENGTH = 240000;

        wave w3 = placeholder(LENGTH, false, false); // Create a waveform of size LENGTH, with one marker
        wave w4 = placeholder(LENGTH, false, false); // Create a waveform of size LENGTH, with one marker


        assignWaveIndex(w3,w4,11);                 // Create a wave table entry with placeholder waveform              
                                                // routed to output 3, with index 10
        // execute the command table entry
        executeTableEntry(0);

        while(true){
        waitDigTrigger(1);
        playWave(w3,w4);
        waitWave();
        };
        """
        # AWG_INDEX = 0 #use channel 1&2 for 2*2
        # awg = device.awgs[AWG_INDEX]
        WAVE_INDEX = 0
        TABLE_INDEX = 0

        # load the sequence
        self.awg.load_sequencer_program(seqc_program)
        self.awg2.load_sequencer_program(seqc_program2)

        # initialize the command table
        ct_schema = self.awg.commandtable.load_validation_schema()
        ct = CommandTable(ct_schema)
        # Wavefrom with amplitude and phase settings
        ct.table[TABLE_INDEX].waveform.index = WAVE_INDEX
        ct.table[TABLE_INDEX].amplitude0.value = 1
        ct.table[TABLE_INDEX].amplitude1.value = 1


        ##Generate a waveform and marker
        Fs = 2400 # in MHz
        nu = self.config['nu_IQ'] # IQ Mod frequency in MHz
        t_mw = int(self.config['count_t'] / 1000) # MW duration converted to us
        A = self.config['IQ_amplitude'] # amplitude
        B = self.config['IQ_amplitude_2'] # amplitude
        phase_shift = self.config['phase_shift'] # phase shift in degree

        LENGTH = t_mw * Fs
        # LENGTH = 240000
        # number of points
        wave1 = A*np.sin(np.linspace(0, 2*np.pi*nu*t_mw, LENGTH))
        wave2 = A*np.cos(np.linspace(0, 2*np.pi*nu*t_mw, LENGTH))
        wave3 = B*np.sin(np.linspace(0, 2*np.pi*nu*t_mw + np.deg2rad(phase_shift), LENGTH))
        wave4 = B*np.cos(np.linspace(0, 2*np.pi*nu*t_mw + np.deg2rad(phase_shift), LENGTH))
        # marker = np.concatenate([np.ones(32), np.zeros(LENGTH-32)]).astype(int)

        ## Upload waveforms``

        print(seqc_program)
        print(seqc_program2)

        waveforms = Waveforms()
        waveforms2 = Waveforms()
        waveforms[10] = (wave1,wave2,None) # I-component (output channel 1), Q-component (output channel 2), marker
        waveforms2[11] = (wave3,wave4,None) # I-component (output channel 1), Q-component (output channel 2), marker
        # awg.commandtable.upload_to_device(ct)
        # awg2.commandtable.upload_to_device(ct2)
        self.awg.write_to_waveform_memory(waveforms)
        self.awg2.write_to_waveform_memory(waveforms2)
        self.awg.enable_sequencer(single=True)
        self.awg2.enable_sequencer(single=True)
        # self.awg.enable_sequencer(single = False)
        # self.awg2.enable_sequencer(single=False)
