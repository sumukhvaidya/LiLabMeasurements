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
import mdt69x
import time
import keyboard


class confocal_Microscope:
    def __init__(self, reset=True, parameters=None):
        if parameters is None:
            parameters = {
                'samples_per_axis': 21,
                'scale': 4,  # volts per pixel
                'count_time (ms)': 10,
                'LED': False
            }
        bitfile_path =r'C:\Users\Li_Lab_B12\Desktop\NDController-ver2\FPGA Bitfiles\everythingdaq_FPGATarget2_FPGAConfocalver5_EY5LFETs4tQ.lvbitx'
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
        self.get_counts(parameters['samples_per_axis']//2, parameters['samples_per_axis']//2, parameters)
        

    def get_counts(self, x, y, parameters):
        xy_volts = np.array([
            (x - parameters['samples_per_axis'] // 2) * parameters['scale'],
            (y - parameters['samples_per_axis'] // 2) * parameters['scale']
        ])
        count_ticks = 40000 * parameters['count_time (ms)']

        self.session.registers['Full Loop Timer(Ticks)'].write(count_ticks + 12 * 40000)
        self.session.registers['AO4'].write(16500 if parameters['LED'] else 0)
        self.session.registers['WaitTime(ticks)'].write(10 * 40000)
        self.session.registers['Counting Time (tick)'].write(count_ticks)

        self.host2target.write(xy_volts)
        self.host2target.start()

        count = self.target2host.read(number_of_elements=1, timeout_ms=-1)
        return count.data[0]
    
    def perform_2d_scan(self, parameters):
        # Initialize a 101x101 array with zeros
        scan_data = np.zeros((parameters['samples_per_axis'],parameters['samples_per_axis']))

        # set colormap
        # r'C:\Users\Li_Lab_B12\Desktop\DataSumukh\250731_PythonCode\colormap_LiLab.csv'
        LiLabColormap =matplotlib.colors.ListedColormap(np.loadtxt('C:\\Users\\Li_Lab_B12\\Desktop\\DataSumukh\\250731_PythonCode\\Main\\colormap_LiLab.csv', delimiter=','), name='Lilab', N=None)

        # Perform the scan row by row
        self.get_counts(parameters['samples_per_axis']//2, parameters['samples_per_axis']//2, parameters) # Run this because the LED flashes briefly and will spoil data.
        time.sleep(0.05)
        for x in tqdm(range(parameters['samples_per_axis'])):
            for y in range(parameters['samples_per_axis']):
                scan_data[x, y] = self.get_counts(x, y, parameters)  # Update the array with scanned data
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(12,10))
            im = ax.imshow(scan_data, cmap=LiLabColormap, interpolation='nearest')
            fig.colorbar(im, label='Counts')
            plt.pause(0.01)
            
        self.get_counts(parameters['samples_per_axis']//2, parameters['samples_per_axis']//2, parameters)  # Final call to bring focus back to center
        return scan_data

    def simple_continuous_counter(self, x, y, parameters):
        """
        A simple counter function that returns the count in time for a specific x, y position.
        """
        if not parameters['window_size']: parameters['window_size'] = 100  # Default window size if not provided
        i = 0
        xdata, ydata = [],[]
        while True:
        # Perform your task here (e.g., scanning or processing)
            xdata.append(i)
            ydata.append(self.get_counts(x, y, parameters))
            if len(xdata) > parameters['window_size']:
                xdata = xdata[-parameters['window_size']:]
                ydata = ydata[-parameters['window_size']:]

            text1 = "Counts = "+str(ydata[-1])+" Running... Press Q to STOP"
        
            fig, ax = plt.subplots(1, 2, figsize=(16,6))
            clear_output(wait=True)
            ax[0].set_title(text1, fontsize=20)
            ax[0].plot(xdata, ydata, marker='o', markersize=2, linestyle='-', linewidth=2, color='blue')
            ax[0].tick_params(axis='both', which='major', labelsize=14)
            ax[0].set_xlabel('Time (arb)', fontsize=16)
            ax[0].set_ylabel('Counts', fontsize=16)
            ax[1].text(0.5, 0.5, 'Counts = \n'+str(ydata[-1]), fontsize=80, ha='center', va='center')
            ax[1].axis('off')
            plt.show()
            plt.pause(0.01)
            # Wait for user input

            if keyboard.is_pressed('q'):
                print("Terminating loop...")
                break
            i += 1


# FInish this autofocus code!
    def autofocus(self, parameters, instruments):
        """
        A simple autofocus function that uses the confocal microscope to find the best focus.
        """
        if not parameters['window_size']: parameters['window_size'] = 100

    def close(self):
        self.session.close()


if __name__ == "__main__":
    None
    