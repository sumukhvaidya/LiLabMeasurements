
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
import mdt69x as mdt69x
import time
import keyboard

LiLabColormap =matplotlib.colors.ListedColormap(np.loadtxt('C:\\Users\\LiLabDesktop\\Desktop\\Sumukh\\LiLabMeasurements\\LiLabMeasurements\\Main\\colormap_LiLab.csv', delimiter=','), name='Lilab', N=None)


def plot_this_xy(xdata, ydata, ax, title, xlabel, ylabel, linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2):
    ax.set_title(title, fontsize=20)
    ax.plot(xdata, ydata, linestyle = linestyle, color = color, linewidth = linewidth, marker = marker, markersize = markersize)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

class ODMR:
    def __init__(self, config, ps=None, sg386=None, rm=None):
        """
        Initialize ODMR class with configuration and instrument handles
        
        Args:
            config (dict): Configuration dictionary with measurement parameters
            ps: PulseStreamer instance (optional)
            sg386: SG386 signal generator instance (optional) 
            rm: PyVISA ResourceManager instance (optional)
        """
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
    
    def create_pulse_sequence(self, aomvolt=None, pulse_sequence=None):
        """Create pulse sequence for ODMR measurement"""
        self.validate_instruments()
        
        aomvolt = aomvolt or self.config['aomvolt']
        pulse_sequence = pulse_sequence or self.config['pulse_sequence']
        
        # Create pulse sequence. Default is CW ODMR
        if pulse_sequence is None or pulse_sequence == 'cw_odmr':
            # Ch0: Laser
            ch0patt = [(self.config['wait_t'], 1), 
                    (self.config['count_t'], 1), 
                    (self.config['addl_t'], 1),
                    (self.config['wait_t'], 1), 
                    (self.config['count_t'], 1), 
                    (self.config['addl_t'], 1),
                    (self.config['separation_t'], 1)]  # Laser always ON
            # CH1: MW 
            ch1patt = [(self.config['wait_t'], 0), 
                    (self.config['count_t'], 0), 
                    (self.config['addl_t'], 0),
                    (self.config['wait_t'], 1), 
                    (self.config['count_t'], 1), 
                    (self.config['addl_t'], 1),
                    (self.config['separation_t'], 0)]
            # CH2: Counter
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
        
        if pulse_sequence == 'rabi':
            # Ch0: Laser
            ch0patt = [(self.config['init_t'], 1), 
                    (self.config['wait_t'], 0), 
                    (self.config['delay_t'], 1),
                    (self.config['readout_t'], 1), 
                    (self.config['init_t'], 1), 
                    (self.config['wait_t'], 0), 
                    (self.config['delay_t'], 1),
                    (self.config['readout_t'], 1),]
            # CH1: MW
            ch1patt = [(self.config['init_t']+self.config['wait_t']+self.config['delay_t']+self.config['readout_t'], 0),
                    (self.config['init_t']+self.config['wait_t']+self.config['delay_t']-self.config['extra_delay']-self.config['tau_set'], 0), 
                    (self.config['tau_set'], 1),
                    (self.config['readout_t']+self.config['extra_delay'], 0),]
            # CH2: Counter
            ch2patt =  [(self.config['init_t'], 0), 
                    (self.config['wait_t'], 0), 
                    (self.config['delay_t'], 0),
                    (self.config['readout_t'], 1), 
                    (self.config['init_t'], 0), 
                    (self.config['wait_t'], 0), 
                    (self.config['delay_t'], 0),
                    (self.config['readout_t'], 1),] 
            seq = self.ps.createSequence()
            seq.setDigital(0, ch0patt)
            seq.setDigital(1, ch1patt)
            seq.setDigital(2, ch2patt)

        return seq
    
    def get_contrast(self, freq=None, plot_sequence=False, config = None):
        """Get ODMR contrast at specified frequency"""
        self.validate_instruments()
        config = config or self.config

        freq = freq or config['freq']

        # Update frequency
        self.sg386.write(f'FREQ {freq}')
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
            target2host.configure(config['pulsenum'])
            target2host.start()


            self.ps.stream(seq, n_runs=(config['pulsenum'])//2) # Run the experiment
            read_value = target2host.read(config['pulsenum'], 1000) # Read the counts as an array
            # read_value = target2host.read(config['pulsenum']*2, 1000) # Read the counts as an array
            target2host.stop() # Stop the register
            
            counts = read_value[0]
            # contrast = (sum(counts[0::2]) * (self.config['pulsenum']//2) / (sum(counts[1::2])*(self.config['pulsenum']//2+1)))
            # contrast = sum(counts[1::2]) / sum(counts[0::2])
        count_off = sum(counts[2::2])
        count_on = sum(counts[3::2])   
        contrast = count_on / count_off 
        # return contrast, sum(counts[2::2])*1e9/(len(counts)*self.config['count_t']), sum(counts[3::2])*1e9/(len(counts)*self.config['count_t'])
        return contrast, count_on, count_off
    
    def run_frequency_sweep(self, freq_range, plot_sequence_first=True):
        """Run ODMR measurement over a frequency range"""
        self.validate_instruments()
        
        peak_trial = self.config['freq']
        
        all_contrast = np.zeros((self.config['num_avgs'],len(freq_range)))
        avg_contrast = np.zeros(len(freq_range))
        all_counts = np.zeros((self.config['num_avgs'],len(freq_range)))
        avg_counts = np.zeros(len(freq_range))
        
        
        for j in tqdm(range(self.config['num_avgs'])):
            contrast = np.zeros(len(freq_range))
            mw_on = np.zeros(len(freq_range))
            mw_off = np.zeros(len(freq_range))

            if self.config['ODMR_Peakfinder'] and j%self.config['peakfinder_check_step']==0:
                self.peakfinder_ODMR(plotting=False, freq = peak_trial)

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
    
    def run_rabi_sweep(self, tau_range, plot_sequence_first=True):
        """Run ODMR measurement over a Tau range"""
        self.validate_instruments()


        all_contrast = np.zeros((self.config['num_avgs'],len(tau_range)))
        avg_contrast = np.zeros(len(tau_range))
        all_counts = np.zeros((self.config['num_avgs'],len(tau_range)))
        avg_counts = np.zeros(len(tau_range))


        for j in tqdm(range(self.config['num_avgs'])):
            contrast = np.zeros(len(tau_range))
            mw_on = np.zeros(len(tau_range))
            mw_off = np.zeros(len(tau_range))

            # for i, freq in enumerate(tqdm(freq_range, desc="Frequency sweep")):
            for i, tau in enumerate(tau_range):
                self.config['tau_set'] = tau
                plot_seq = plot_sequence_first and i == 0
                
                if keyboard.is_pressed('q'):
                    print(f"\nMeasurement stopped by user at tau = {tau:.2e} ns")
                    # Return partial results up to current point
                    return tau_range, avg_contrast, all_contrast, avg_counts, all_counts

                try:
                    contrast[i], mw_on[i], mw_off[i] = self.get_contrast( plot_sequence=plot_seq)
                    
                    fig, ax = plt.subplots(2, 2, figsize=(16,12))
                    clear_output(wait=True)
                    fig.suptitle(f'Rabi Measurement - Run {j+1}/{self.config["num_avgs"]}. Press Q to stop', fontsize=22, y=0.95)

                    plot_this_xy(tau_range[:i], contrast[:i], ax = ax[0][0], title = 'Current Run Rabi Contrast', xlabel = 'Tau (ns)', ylabel = 'Contrast', linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2)
                    plot_this_xy(tau_range, avg_contrast, ax = ax[0][1], title = 'Average Rabi Contrast', xlabel = 'Tau (ns)', ylabel = 'Average Contrast', linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2)

                    plot_this_xy(tau_range[:i], mw_on[:i]+mw_off[:i], ax = ax[1][0], title = 'Average Rabi Counts', xlabel = 'Tau (ns)', ylabel = 'Average Counts', linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2)
                    plot_this_xy(tau_range, avg_counts, ax = ax[1][1], title = 'Average Rabi Counts', xlabel = 'Tau (ns)', ylabel = 'Average Counts', linestyle = '-', color = 'blue', linewidth = 2, marker = 'o', markersize = 2)

                    plt.show()
                    plt.pause(0.01)
                except Exception as e:
                    print(f"Error at tau {tau}: {e}")
                    continue
            avg_contrast = (avg_contrast*(j)+contrast)/(j+1)
            all_contrast[j,:] = contrast
            avg_counts = (avg_counts*(j)+mw_on+mw_off)/(j+1)
            all_counts[j,:] = mw_on+mw_off
            time.sleep(0.5)

        return tau_range, avg_contrast, all_contrast, avg_counts, all_counts

    def peakfinder_ODMR(self, config = None, plotting=True, freq = None):
        if not config:
            config = self.config
        
        window = config['peakfinder_window']
        center = config['position']
        contrasts = np.ones([window, window])
        for i in range(window):
            for j in range(window):
                config['position'] = [center[0]+i-window//2, center[1]+j-window//2]
                contrasts[i][j],_,_ = self.get_contrast( freq)
    
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
        print('Best Position = ', config['position'])
        time.sleep(.5)

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