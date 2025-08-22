# import numpy as np
# import time
# import scipy
# import matplotlib
# import matplotlib.pyplot as plt
# import pulsestreamer
# import nifpga
# import pyvisa
# import serial
# import io
# import dev.systemFunctions as systemFunctions
# from IPython.display import clear_output
# from tqdm import tqdm
# import sys
# import mdt69x
# import time
# import keyboard

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
    
    def get_contrast(self, freq=None, plot_sequence=False):
        """Get ODMR contrast at specified frequency"""
        self.validate_instruments()
        
        freq = freq or self.config['freq']
        
        # Update frequency
        self.sg386.write(f'FREQ {freq}')
        self.sg386.write('AMPR '+str(self.config['mw_power']) )
        
        # Create pulse sequence
        seq = self.create_pulse_sequence()
        if plot_sequence:
            # seq.plot()
            pass
        
        # Connect to FPGA and run measurement
        bitfile_loc = r"C:\Users\Li_Lab_B12\Desktop\DataSumukh\250731_PythonCode\bitfiles\everythingdaq_FPGATarget2_FPGAESRver5_RELKYtnkXk4.lvbitx"
        
        with nifpga.Session(bitfile=bitfile_loc, resource="RIO0") as session:
            session.reset()
            session.run()
            target2host = session.fifos['FIFO_target2host']
            target2host.configure(self.config['pulsenum'])
            target2host.start()
            
            self.ps.stream(seq, n_runs=int(self.config['pulsenum']/2))
            read_value = target2host.read(self.config['pulsenum'], 1000)
            target2host.stop()
            
            counts = read_value[0]
            contrast = sum(counts[0::2]) / sum(counts[1::2])
            
        return contrast, sum(counts[0::2])*1e9/(len(counts)*self.config['count_t']), sum(counts[1::2])*1e9/(len(counts)*self.config['count_t'])
    
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