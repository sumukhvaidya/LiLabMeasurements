import matplotlib
from IPython.display import clear_output
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
import pulsestreamer
import nifpga
import pyvisa
import serial
import io
import sys
import mdt69x as mdt69x
import time
import keyboard
import pulsestreamer



class confocal_Microscope:
    def __init__(self, reset=True, config=None, ps=None):
        if config is None:
            self.config = {
                'samples_per_axis': 21,
                'scale': 4,  # volts per pixel
                'count_time (ms)': 10,
                'LED': False,
            }
        else:
            self.config = config
        self.ps = ps 
        # bitfile_path =r'C:\Users\Li_Lab_B12\Desktop\NDController-ver2\FPGA Bitfiles\everythingdaq_FPGATarget2_FPGAConfocalver5_EY5LFETs4tQ.lvbitx'
        bitfile_path =r'C:\Users\LiLabDesktop\Desktop\Sumukh\LiLabMeasurements\LiLabMeasurements\bitfiles\everythingdaq_FPGATarget2_FPGAConfocalver5_EY5LFETs4tQ.lvbitx'
        self.session = nifpga.Session(bitfile=bitfile_path, resource='RIO0')
        if reset:
            self.session.reset()
            self.session.run()
        else:
            self.session.run()
        self.host2target = self.session.fifos['FIFO_Host2Target']
        self.host2target.configure(10)
        self.target2host = self.session.fifos['FIFO_target2host']
        self.target2host.configure(10)
        self.get_counts(self.config['samples_per_axis']//2, self.config['samples_per_axis']//2, self.config)

    def get_counts(self, x, y, config):
        xy_volts = np.array([
            (x - config['samples_per_axis'] // 2) * config['scale'],
            (y - config['samples_per_axis'] // 2) * config['scale']
        ])
        count_ticks = 40000 * config['count_time (ms)']

        self.session.registers['Full Loop Timer(Ticks)'].write(count_ticks + 12 * 40000)
        self.session.registers['AO4'].write(16500 if config['LED'] else 0)
        self.session.registers['WaitTime(ticks)'].write(10 * 40000)
        self.session.registers['Counting Time (tick)'].write(count_ticks)

        self.host2target.write(xy_volts)
        self.host2target.start()

        count = self.target2host.read(number_of_elements=1, timeout_ms=-1)
        return count.data[0]
    
    def perform_2d_scan(self, config):
        # Initialize a 101x101 array with zeros
        scan_data = np.zeros((config['samples_per_axis'], config['samples_per_axis']))

        # set colormap
        # r'C:\Users\Li_Lab_B12\Desktop\DataSumukh\250731_PythonCode\colormap_LiLab.csv'
        LiLabColormap =matplotlib.colors.ListedColormap(np.loadtxt('C:\\Users\\LiLabDesktop\\Desktop\\Sumukh\\LiLabMeasurements\\LiLabMeasurements\\Main\\colormap_LiLab.csv', delimiter=','), name='Lilab', N=None)

        # Perform the scan row by row
        self.get_counts(config['samples_per_axis']//2, config['samples_per_axis']//2, config) # Run this because the LED flashes briefly and will spoil data.
        time.sleep(0.05)
        self.laser_on()
        for x in tqdm(range(config['samples_per_axis'])):
            for y in range(config['samples_per_axis']):
                scan_data[x, y] = self.get_counts(x, y, config)  # Update the array with scanned data
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(12,10))
            im = ax.imshow(scan_data, cmap=LiLabColormap, interpolation='nearest')
            fig.colorbar(im, label='Counts')
            fig.suptitle(f'Running... Press: Q = STOP. L = Toggle LED', fontsize=22, y=0.95)
            plt.pause(0.01)
            if keyboard.is_pressed('q'):
                print("Terminating loop...")
                break

            if keyboard.is_pressed('l'):
                config['LED'] = not config['LED']

        self.get_counts(config['samples_per_axis']//2, config['samples_per_axis']//2, config)  # Final call to bring focus back to center
        self.laser_off()
        return scan_data

    def simple_continuous_counter(self, x, y, config):
        """
        A simple counter function that returns the count in time for a specific x, y position.
        """
        if not config['window_size']: config['window_size'] = 100  # Default window size if not provided
        i = 0
        xdata, ydata = [],[]
        while True:
        # Perform your task here (e.g., scanning or processing)
            xdata.append(i)
            ydata.append(self.get_counts(x, y, config))
            if len(xdata) > config['window_size']:
                xdata = xdata[-config['window_size']:]
                ydata = ydata[-config['window_size']:]

            text1 = " Running... Press: Q = STOP. L = Toggle LED"

            fig, ax = plt.subplots(1, 2, figsize=(16,6))
            clear_output(wait=True)
            # ax[0].set_title('Counts', fontsize=20)
            ax[0].plot(xdata, ydata, marker='o', markersize=2, linestyle='-', linewidth=2, color='blue')
            ax[0].tick_params(axis='both', which='major', labelsize=14)
            ax[0].set_xlabel('Time (arb)', fontsize=16)
            ax[0].set_ylabel('Counts', fontsize=16)
            ax[1].text(0.5, 0.5, 'Counts = \n'+str(ydata[-1]), fontsize=80, ha='center', va='center')
            ax[1].axis('off')
            fig.suptitle(text1, fontsize=22, y=0.95)
            plt.show()
            plt.pause(0.01)
            # Wait for user input

            if keyboard.is_pressed('q'):
                print("Terminating loop...")
                break

            if keyboard.is_pressed('l'):
                config['LED'] = not config['LED']
                
            i += 1


    def toggle_LED(self):
        """Toggle the LED state."""
        self.config['LED'] = not self.config['LED']
        if self.config['LED']:
            self.session.registers['AO4'].write(16500)
        else:
            self.session.registers['AO4'].write(0)
        return
    def laser_on(self):
        """Turn the laser on."""
        self.ps.constant(([0], 0, 0))
        return
    
    def laser_off(self):
        """Turn the laser off."""
        self.ps.constant(([], 0, 0))
        return
# Finish this autofocus code!
    def autofocus(self, ):
        """
        A simple autofocus function that uses the confocal microscope to find the best focus.
        """
        if not config['window_size']: config['window_size'] = 100

    def close(self):
        self.session.close()
    
    def reset(self):
        self.session.reset()

    def set_position(self, x, y, config):
        """
        Set the position of the confocal microscope.
        """
        xy_volts = np.array([
            (x - config['samples_per_axis'] // 2) * config['scale'],
            (y - config['samples_per_axis'] // 2) * config['scale']
        ])
        self.host2target.write(xy_volts)
        return


if __name__ == "__main__":
    None
    