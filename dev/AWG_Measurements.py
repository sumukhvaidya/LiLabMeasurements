
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


# class AWGSetup:
#     def __init__(self, config, awg):
#         self.config = config
#         self.awg = awg

#     def Rabi_2_channel_setup(self):
#         """Run Rabi measurement over a tau range"""
#         ############ AWG setup ############
#         # sequence parameters
#         nu = self.config['nu_IQ'] #385 #385 #100 #100#388#385.7 # microwave frequency in MHz
#         Fs = 2400 
#         NUM_SUBCYCLES = self.config['n_points']  # Number of subcycles #Number of points
#         tau_step0 =self.config['tau_step'] # tau step in us
#         tau_start0 = self.config['tau_start'] # starting tau in us
#         # t_tot = 78 # total time of a pulse frame in us
#         t_tot = self.config['init_t'] + self.config['readout_t'] + self.config['wait_t'] + self.config['delay_t'] # total time of a pulse frame in us
#         TT_tot = t_tot*NUM_SUBCYCLES*2  # total time need for measuring all data points once
#         t_initi = self.config['init_t'] #laser initialization time
#         A = self.config['IQ_amplitude'] # 0.65#0.5087 #0.6 #0.3393#0.3#0.7
#         # A = 0.36 # 100 T=35ns
#         tau_range = np.linspace(tau_start0, tau_start0 + tau_step0 * (NUM_SUBCYCLES - 1), NUM_SUBCYCLES)  # tau range in us 

#         # convert parameters
#         TAU_STEP = int(tau_step0*Fs) # convert tau step into tick
#         #TAU_START = int(tau_start0*Fs) # convert tau step into tick
#         DATA_POINTS_PER_SUBCYCLE = int(t_tot*Fs)  # Data points per subcycle
#         DATA_POINTS_TOTAL = DATA_POINTS_PER_SUBCYCLE*NUM_SUBCYCLES*2 # total waveform points for measuring all data points once
#         INITIAL_PERIOD_LENGTH =  int(tau_start0*Fs)  # Initial tau
#         MAX_PERIOD_LENGTH = INITIAL_PERIOD_LENGTH + (NUM_SUBCYCLES - 1) * TAU_STEP  # Maximum length of tau
#         WFM_LEN = NUM_SUBCYCLES * DATA_POINTS_PER_SUBCYCLE*2  # Total waveform length
#         T_INITIAL = int((t_initi+0.12)*Fs)


#         # define sequence veriable and send it the sequencer in LabOne

#         seqc_a = """\
#         const LENGTH = """
#         seqc_b = str(DATA_POINTS_TOTAL)
#         seqc_c = """; // Number of subcycles * Data points per subcycle

#         wave w1 = placeholder(LENGTH, false, false); // Create a waveform of size LENGTH, with one marker
#         wave w2 = placeholder(LENGTH, false, false); // Create a waveform of size LENGTH, with one marker

#         assignWaveIndex(w1,w2,10);                 // Create a wave table entry with placeholder waveform              
#                                                 // routed to output 1, with index 10

#         // execute the command table entry
#         executeTableEntry(0);

#         while(true){
#         waitDigTrigger(1);
#         playWave(w1,w2);

#         waitWave();
#         };
#         """

#         seqc_program = seqc_a + seqc_b +seqc_c
#         # seqc_program = seqc_b
#         print(seqc_program)

        
#         WAVE_INDEX = 0
#         TABLE_INDEX = 0

#         # load the sequence
#         self.awg.load_sequencer_program(seqc_program)

#         wave1 = np.zeros(WFM_LEN)
#         wave2 = np.zeros(WFM_LEN)

#         # Generate sine waves for each subcycle
#         for subcycle in range(NUM_SUBCYCLES):
#             current_length = INITIAL_PERIOD_LENGTH + subcycle * TAU_STEP  # Length for the current sine wave
#             start_index = subcycle * DATA_POINTS_PER_SUBCYCLE*2 + T_INITIAL # Center position
            
#             # Generate sine wave
#             for i in range(current_length):
#                 wave1[start_index + i] = A*np.sin( 2*np.pi*i*nu/Fs)  # Sine wave calculation

#         # Generate cosine waves for each subcycle
#         for subcycle in range(NUM_SUBCYCLES):
#             current_length = INITIAL_PERIOD_LENGTH + subcycle * TAU_STEP  # Length for the current sine wave
#             start_index = subcycle * DATA_POINTS_PER_SUBCYCLE*2 + T_INITIAL# Center position
            
#             # Generate sine wave
#             for i in range(current_length):
#                 wave2[start_index + i] = A*np.cos( 2*np.pi*i*nu/Fs)  # Sine wave calculation

#         ## Upload waveforms

#         waveforms = Waveforms()
#         waveforms[10] = (wave1,wave2,None) # I-component (output channel 1), Q-component (output channel 2), marker
#         # awg.commandtable.upload_to_device(ct)
#         self.awg.write_to_waveform_memory(waveforms)
#         self.awg.enable_sequencer(single=True)


class AWGMeasurement:
    def __init__(self, config, ps=None, sg386=None, awg=None, rm=None):
        """
        Initialize AWGMeasurement class with configuration and instrument handles

        Args:
            config (dict): Configuration dictionary with measurement parameters
            ps: PulseStreamer instance (optional)
            sg386: SG386 signal generator instance (optional) 
            rm: PyVISA ResourceManager instance (optional)
        """
        self.config = config
        self.ps = ps
        self.sg386 = sg386
        self.awg = awg
        self.rm = rm
        
        # Validate that required instruments are provided
        if ps is None or sg386 is None or awg is None:
            print("Warning: Some instruments not provided. Use set_instruments() to set them later.")

    def set_instruments(self, ps, sg386, awg, rm=None):
        """Set instrument handles after initialization"""
        self.ps = ps
        self.sg386 = sg386
        self.qwg = awg
        self.rm = rm
    
    def validate_instruments(self):
        """Check if all required instruments are available"""
        if self.ps is None:
            raise RuntimeError("PulseStreamer not set. Call set_instruments() or pass ps to __init__")
        if self.sg386 is None:
            raise RuntimeError("SG386 not set. Call set_instruments() or pass sg386 to __init__")
        if self.awg is None:
            raise RuntimeError("AWG not set. Call set_instruments() or pass awg to __init__")

    def create_pulse_sequence(self, aomvolt=None, pulse_sequence=None):
        """Create pulse sequence for AWG measurement"""
        self.validate_instruments()
        
        aomvolt = aomvolt or self.config['aomvolt']
        pulse_sequence = pulse_sequence or self.config['pulse_sequence']
        
        # Create pulse sequence. Default is CW ODMR
        if pulse_sequence is None or pulse_sequence == 'cw_odmr':
            # Ch0: Laser
            ch0patt = [(self.config['wait_t'] , 1), 
                    (self.config['count_t'] , 1), 
                    (self.config['addl_t'], 1),
                    (self.config['wait_t'] , 1), 
                    (self.config['count_t'] , 1), 
                    (self.config['addl_t'] , 1),
                    (self.config['separation_t'] , 1)]  # Laser always ON
            # CH1: MW 
            ch1patt = [(self.config['wait_t'] , 1), 
                    (self.config['count_t'] , 1), 
                    (self.config['addl_t'] , 1),
                    (self.config['wait_t'] , 1), 
                    (self.config['count_t'] , 1), 
                    (self.config['addl_t'] , 1),
                    (self.config['separation_t'] , 1)]
            # CH2: Counter
            ch2patt = [(self.config['wait_t'] , 0), 
                    (self.config['count_t'] , 1), 
                    (self.config['addl_t'] , 0),
                    (self.config['wait_t'] , 0), 
                    (self.config['count_t'] , 1), 
                    (self.config['addl_t'] , 0),
                    (self.config['separation_t'] , 0)]
            ch3patt = [(50, 1), 
                    (self.config['wait_t']-50, 0), 
                    (self.config['count_t'], 0),
                    (self.config['addl_t'], 0), 
                    (self.config['wait_t'], 0),
                    (self.config['count_t'], 0), 
                    (self.config['addl_t'], 0),
                    (self.config['separation_t'], 0),] 

            seq = self.ps.createSequence()
            seq.setDigital(0, ch0patt)
            seq.setDigital(1, ch1patt)
            seq.setDigital(2, ch2patt)
            seq.setDigital(3, ch3patt)

        if pulse_sequence == 'rabi':
            # Ch0: Laser
            ch0patt = [(self.config['init_t'] * 1e3, 1), 
                    (self.config['wait_t'] * 1e3, 0), 
                    (self.config['delay_t'] * 1e3, 1),
                    (self.config['readout_t'] * 1e3, 1), 
                    (self.config['init_t'] * 1e3, 1), 
                    (self.config['wait_t'] * 1e3, 0), 
                    (self.config['delay_t'] * 1e3, 1),
                    (self.config['readout_t'] * 1e3, 1),]
            # CH1: MW
            ch1patt =  [(self.config['init_t'] * 1e3, 1), 
                    (self.config['wait_t'] * 1e3, 1), 
                    (self.config['delay_t'] * 1e3, 1),
                    (self.config['readout_t'] * 1e3, 1), 
                    (self.config['init_t'] * 1e3, 1),
                    (self.config['wait_t'] * 1e3, 1), 
                    (self.config['delay_t'] * 1e3, 1),
                    (self.config['readout_t'] * 1e3, 1),]
            # CH2: Counter
            ch2patt =  [(self.config['init_t'] * 1e3, 0), 
                    (self.config['wait_t'] * 1e3, 0), 
                    (self.config['delay_t'] * 1e3, 0),
                    (self.config['readout_t'] * 1e3, 1), 
                    (self.config['init_t'] * 1e3, 0), 
                    (self.config['wait_t'] * 1e3, 0), 
                    (self.config['delay_t'] * 1e3, 0),
                    (self.config['readout_t'] * 1e3, 1),] 
            # CH3:  AWG triggering Channel
            ch3patt =  [(104, 1), 
                    (self.config['wait_t']*1e3-104, 0), 
                    (self.config['delay_t'] * 1e3, 0),
                    (self.config['readout_t'] * 1e3, 0), 
                    (self.config['init_t'] * 1e3, 0), 
                    (self.config['wait_t'] * 1e3, 0), 
                    (self.config['delay_t'] * 1e3, 0),
                    (self.config['readout_t'] * 1e3, 0),] 
            seq = self.ps.createSequence()
            seq.setDigital(0, ch0patt)
            seq.setDigital(1, ch1patt)
            seq.setDigital(2, ch2patt)
            seq.setDigital(3, ch3patt)

            seq = seq * self.config['n_points']
        return seq
    
    def get_contrast(self, freq=None, plot_sequence=False, config = None):
        """Get ODMR contrast at specified frequency"""
        self.validate_instruments()
        config = config or self.config

        freq = freq or config['freq']

        # Update frequency
        # self.sg386.write('MODL 1' ) # Turn on IQ Modulation on SG386
        self.sg386.write('MODL 0' ) # Turn off IQ Modulation on SG386
        self.sg386.write(f'FREQ {freq + config["nu_IQ"]*1e6}' ) # Set carrier frequency, in HZ
        self.sg386.write('AMPR '+str(config['mw_power']) )

        # Create pulse sequence
        seq = self.create_pulse_sequence(pulse_sequence=self.config['pulse_sequence'])
        if plot_sequence:
            # seq.plot()
            pass
        
        # Connect to FPGA and run measurement
        bitfile_loc = r"C:\Users\LiLabDesktop\Desktop\Sumukh\LiLabMeasurements\LiLabMeasurements\bitfiles\everythingdaq_FPGATarget2_FPGAESRver5_RELKYtnkXk4.lvbitx"
            
        with nifpga.Session(bitfile=bitfile_loc, resource="RIO0") as session:
            session.reset()
            session.run()

            # Calculate x, y voltage for fixing position
            xy_volts = np.array([
            (config['position'][0] - config['samples_per_axis'] // 2) * config['scale'],
            (config['position'][1] - config['samples_per_axis'] // 2) * config['scale']])
             # Write the position to the FPGA
            # host2target = session.fifos['FIFO_Host2Target']
            # host2target.write(xy_volts)
            session.registers['x'].write(xy_volts[0]) 
            session.registers['y'].write(xy_volts[1]) 

            # Setup readout of the counts
            target2host = session.fifos['FIFO_target2host']
            # target2host.configure(config['pulsenum'])
            target2host.configure(10000)
            target2host.start()


            self.ps.stream(seq, n_runs=(config['pulsenum'])) # Run the experiment
            read_value = target2host.read(config['pulsenum']*2, 4000) # Read the counts as an array
            # read_value = target2host.read(config['pulsenum']*2, 1000) # Read the counts as an array
            target2host.stop() # Stop the register
            
            counts = read_value[0]
            # implement readout logic here
            # contrast = (sum(counts[0::2]) * (self.config['pulsenum']//2) / (sum(counts[1::2])*(self.config['pulsenum']//2+1)))
            # contrast = sum(counts[3::2]) / sum(counts[2::2])

        count_off = sum(counts[3::2])
        count_on = sum(counts[2::2])   
        contrast = count_on / count_off 
        # return contrast, sum(counts[2::2])*1e9/(len(counts)*self.config['count_t']), sum(counts[3::2])*1e9/(len(counts)*self.config['count_t'])
        return contrast, count_on, count_off   
        # return contrast, sum(counts[0::2])*1e9/(len(counts)*self.config['count_t']), sum(counts[1::2])*1e9/(len(counts)*self.config['count_t'])
    
    def run_frequency_sweep(self, freq_range, plot_sequence_first=True):
        """Run ODMR measurement over a frequency range"""
        self.validate_instruments()
        
        
        all_contrast = np.zeros((self.config['num_avgs'],len(freq_range)))
        avg_contrast = np.zeros(len(freq_range))
        all_counts = np.zeros((self.config['num_avgs'],len(freq_range)))
        avg_counts = np.zeros(len(freq_range))
        
        
        for j in tqdm(range(self.config['num_avgs'])):
            contrast = np.zeros(len(freq_range))
            mw_on = np.zeros(len(freq_range))
            mw_off = np.zeros(len(freq_range))

            # for i, freq in enumerate(tqdm(freq_range, desc="Frequency sweep")):
            for i, freq in enumerate(freq_range):
                self.config['freq'] = freq
                plot_seq = plot_sequence_first and i == 0
                
                if keyboard.is_pressed('q'):
                    print(f"\nMeasurement stopped by user at frequency {freq:.2e} Hz")
                    # Return partial results up to current point
                    return freq_range, avg_contrast, all_contrast, avg_counts, all_counts

                try:
                    contrast[i], mw_on[i], mw_off[i] = self.get_contrast(freq, plot_sequence=plot_seq)
                    fig, ax = plt.subplots(2, 2, figsize=(16,12))
                    clear_output(wait=True)
                    fig.suptitle(f'ODMR Measurement - Run {j+1}/{self.config["num_avgs"]}. Press Q to stop', fontsize=22, y=0.95)
                    
                    plot_this_xy(freq_range[:i], contrast[:i], ax = ax[0][0], title = 'Current Run ODMR Contrast', xlabel = 'Frequency (Hz)', ylabel = 'Contrast', linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2)
                    plot_this_xy(freq_range, avg_contrast, ax = ax[0][1], title = 'Average ODMR Contrast', xlabel = 'Frequency (Hz)', ylabel = 'Average Contrast', linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2)
                    
                    plot_this_xy(freq_range[:i], mw_on[:i]+mw_off[:i], ax = ax[1][0], title = 'Average ODMR Counts', xlabel = 'Frequency (Hz)', ylabel = 'Average Counts', linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2)
                    plot_this_xy(freq_range, avg_counts, ax = ax[1][1], title = 'Average ODMR Counts', xlabel = 'Frequency (Hz)', ylabel = 'Average Counts', linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2)

                    plt.show()
                    plt.pause(0.01)
                except Exception as e:
                    print(f"Error at frequency {freq}: {e}")
                    continue
            avg_contrast = (avg_contrast*(j)+contrast)/(j+1)
            all_contrast[j,:] = contrast
            avg_counts = (avg_counts*(j)+mw_on+mw_off)/(j+1)
            all_counts[j,:] = mw_on+mw_off
        
        return freq_range, avg_contrast, all_contrast, avg_counts, all_counts
    
    def run_rabi_sweep(self,  plot_sequence_first=True):
        """Run Rabi measurement over a tau range"""
        self.validate_instruments()

        self.sg386.write('MODL 1' ) # Turn on IQ Modulation on SG386
        self.sg386.write('FREQ '+str(self.config['freq'] + self.config['nu_IQ']*1e6) ) # Set carrier frequency, in HZ
        self.sg386.write('AMPR '+str(self.config['mw_power']))

        tau_range = np.linspace(self.config['tau_start'], self.config['tau_start'] + self.config['tau_step'] * (self.config['n_points'] - 1), self.config['n_points'])  # tau range in us 

        # Initialize the arrays for storing and processing data
        all_contrast = np.zeros((self.config['num_avgs'],len(tau_range)))
        avg_contrast = np.zeros(len(tau_range))
        all_counts = np.zeros((self.config['num_avgs'],len(tau_range)))
        # all_data2 = np.zeros((self.config['num_avgs'],self.config['n_points'] * 2 * self.config['sequence_repetitions']))
        avg_counts = np.zeros(len(tau_range))
        avg_mw_on = np.zeros(len(tau_range))
        avg_mw_off = np.zeros(len(tau_range))


        for j in tqdm(range(self.config['num_avgs'])):
            
            counts = np.zeros(2*self.config['n_points'])
            contrast = np.zeros(self.config['n_points'])
            mw_on = np.zeros(self.config['n_points'])
            mw_off = np.zeros(self.config['n_points'])

            # # for i, freq in enumerate(tqdm(freq_range, desc="Frequency sweep")):
            # for i, tau in enumerate(tau_range):
            #     self.config['tau_set'] = tau
            #     plot_seq = plot_sequence_first and i == 0
                
            if keyboard.is_pressed('q'):
                print(f"\nMeasurement stopped by user at j = {j:.2e} run")
                # Return partial results up to current point
                return tau_range, avg_contrast, all_contrast, avg_counts, all_counts

            try:

           
                seq = self.create_pulse_sequence(pulse_sequence='rabi')
                bitfile_loc = r"C:\Users\LiLabDesktop\Desktop\Sumukh\LiLabMeasurements\LiLabMeasurements\bitfiles\everythingdaq_FPGATarget2_FPGAESRver5_RELKYtnkXk4.lvbitx"
                
                with nifpga.Session(bitfile=bitfile_loc, resource="RIO0") as session:
                    session.reset()
                    session.run()

                    # Calculate x, y voltage for fixing position
                    xy_volts = np.array([
                    (self.config['position'][0] - self.config['samples_per_axis'] // 2) * self.config['scale'],
                    (self.config['position'][1] - self.config['samples_per_axis'] // 2) * self.config['scale']])
                    # Write the position to the FPGA
                    # host2target = session.fifos['FIFO_Host2Target']
                    # host2target.write(xy_volts)
                    session.registers['x'].write(xy_volts[0]) 
                    session.registers['y'].write(xy_volts[1]) 

                    # Setup readout of the counts
                    target2host = session.fifos['FIFO_target2host']
                    target2host.configure(self.config['n_points']*2*self.config['sequence_repetitions'])
                    # target2host.configure(10000)
                    target2host.start()


                    self.ps.stream(seq, n_runs=self.config['sequence_repetitions']) # Run the experiment
                    read_value = target2host.read(self.config['n_points'] * 2 * self.config['sequence_repetitions'], 5000) # Read the counts as an array
                    # read_value = target2host.read(2 * self.config['sequence_repetitions'], 5000) # Read the counts as an array
                    
                    target2host.stop() # Stop the register
                    
                    counts = read_value[0]    
                
                counts = np.array(counts)
                # Process data
                all_data = counts.reshape((self.config['n_points'] * self.config['sequence_repetitions'], 2))
                mw_off = all_data[:,1].reshape(( self.config['sequence_repetitions'], self.config['n_points'],)).sum(axis=0)
                mw_on = all_data[:,0].reshape(( self.config['sequence_repetitions'], self.config['n_points'],)).sum(axis=0)
                # mw_off = np.array(counts[:self.config['n_points']]) # First half no MW
                # mw_on = np.array(counts[self.config['n_points']:]) # Second half with MW
                
                contrast = mw_on/(mw_off)



                fig, ax = plt.subplots(2, 2, figsize=(16,12))
                clear_output(wait=True)
                fig.suptitle(f'Rabi Measurement - Run {j+1}/{self.config["num_avgs"]}. Press Q to stop', fontsize=22, y=0.95)

                plot_this_xy(tau_range, avg_contrast, ax = ax[0][0], title = 'Average Rabi Contrast', xlabel = 'Tau (us)', ylabel = 'Contrast', linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2)
                plot_this_xy(tau_range, avg_counts, ax = ax[0][1], title = 'Average Rabi Counts', xlabel = 'Tau (us)', ylabel = 'Average Counts', linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2)

                plot_this_xy(tau_range, avg_mw_on, ax = ax[1][0], title = 'Average MW ON Counts', xlabel = 'Tau (us)', ylabel = 'Average Counts', linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2)
                plot_this_xy(tau_range, avg_mw_off, ax = ax[1][1], title = 'Average MW OFF Counts', xlabel = 'Tau (us)', ylabel = 'Average Counts', linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2)

                plt.show()
                plt.pause(0.01)

            except Exception as e:
                print(f"Error at j= {j}: {e}")
                continue

            avg_contrast = (avg_contrast*(j)+contrast)/(j+1)
            all_contrast[j,:] = contrast
            avg_counts = (avg_counts*(j)+mw_on+mw_off)/(j+1)
            all_counts[j,:] = mw_on+mw_off
            avg_mw_off = (avg_mw_off*(j)+mw_off)/(j+1)
            avg_mw_on = (avg_mw_on*(j)+mw_on)/(j+1)
            
        return tau_range, avg_contrast, all_contrast, avg_counts, all_counts

    # def peakfinder_ODMR(self, config = None, window = 1, plotting=True):
        if not config:
            config = self.config
        
        center = config['position']
        contrasts = np.ones([window, window])
        for i in range(window):
            for j in range(window):
                config['position'] = [center[0]+i-window//2, center[1]+j-window//2]
                contrasts[i][j],_,_ = self.get_contrast()
    
                if plotting:
                    clear_output(wait=True)
                    fig, ax = plt.subplots(figsize=(12,10))
                    im = ax.imshow(contrasts, cmap=LiLabColormap, interpolation='nearest')
                    fig.colorbar(im, label='Counts')
                    fig.suptitle(f'Running...', fontsize=22, y=0.95)
                    plt.pause(0.01)
                    # plt.imshow(contrasts, cmap=LiLabColormap, extent=(center[0]-window//2, center[0]+window//2, center[1]-window//2, center[1]+window//2))
        ############!!!!!!!!!!!!!!!! get_contrast needs to pass its own config!
        imax, jmax = np.unravel_index(np.argmax(abs(contrasts-np.ones((window, window)))), contrasts.shape)
        config['position'] = [center[0]+imax-window//2, center[1]+jmax-window//2]
        print('Best Contrast = ', self.get_contrast()[0])

        return

    # def set_AWG_IQ_out(self, config):
        return
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        self.config.update(kwargs)
        print(f"Updated config: {kwargs}")
    
    def get_config(self):
        """Get current configuration"""
        return self.config.copy()

# Example usage in notebook:
if __name__ == "__main__":
    # This would typically be in your notebook:
    
    # 1. Initialize instruments at notebook level
    # pulsestreamer_ip = '192.168.0.100'
    # ps = pulsestreamer.PulseStreamer(pulsestreamer_ip)
    # ps.reset()
    
    # rm = pyvisa.ResourceManager()
    # sg386 = rm.open_resource('GPIB0::27::INSTR')
    # sg386.write('ENBR 1')
    
    # 2. Create ODMR instance with instruments
    config = {
        'aomvolt': 0.5,
        'pulsenum': 10000,
        'count_t': 0.05*1e6,
        'separation_t': 0.03*1e6,
        'addl_t': 0.003*1e6,
        'wait_t': 0.003*1e6,
        'freq': 1.85e9,
        'mw_power': -9,
        'seqplot': True
    }
    
    # odmr = ODMR(config, ps=ps, sg386=sg386, rm=rm)
    
    # 3. Use for ODMR measurements
    # freqs = np.linspace(3.2e9, 3.7e9, 51)
    # freq_range, contrast, mw_on, mw_off = odmr.run_frequency_sweep(freqs)
    
    # 4. Reuse instruments for other experiments
    # confocal = Confocal(config_confocal, ps=ps, sg386=sg386)
    # rabi = Rabi(config_rabi, ps=ps, sg386=sg386)
    
    # 5. Close instruments when done (in notebook)
    # sg386.close()
    # rm.close()