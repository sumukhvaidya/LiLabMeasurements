import numpy as np
import scipy
import matplotlib.pyplot as plt
import tqdm
import pulsestreamer
import nifpga
import pyvisa
import serial
import io
import sys
import mdt69x
import time

# Disable standard output of the PulseStreamer Connection:
trap=io.StringIO()
sys.stdout=trap
# class smallTable(mdt69x.Controller,pulsestreamer.PulseStreamer):
#     def __init__(self):
#         mdt69x.Controller.__init__(self)

#     def confocalScan(self,zvolt=30, xinit=0, yinit=0, stepsize=0.2, vrange=10, tcol=10 )
#         file="SimplifiedCounter_2.vi"
#         with nifpga.Session(bitfile=file, resource="RIO0") as session:
        
#         return
#         session.reset()
#         session.run()
#         target2host=session.fifos['FIFO_target2host']
#         target2host.configure(pulsenum)
#         target2host.start()

# everythingdaq_FPGATarget2_SimplifiedCounte_MEAbZCkdKdg



# Development area
    
# file="C:/NDController-ver2/FPGA Bitfiles/EverythingDAQ_FPGATarget2_SimplifiedCounte_dpWr51Ax0nw.lvbitx"

# with nifpga.Session(bitfile=bitfile, resource="RIO0") as session:
#     # session.reset()
#     # session.run()
#     tcol=session.registers['Count(Ticks)']
#     tcol.write(100000000)
#     fifoin=session.fifos['FIFO_host2target']    
#     fifoin.configure(100)
#     fifoin.start()
#     fifoin.write([0,1])
#     # buff=session.fifos['FIFO_host2target']
#     # buff.configure(1000)
#     time.sleep(1)
#     count=session.registers['Counts']
#     countout=count.read()
#     print(countout)
#     # session





#### ODMR contrast function definitions
def getODMRContrast(config):
    bitfile_loc="C:/NDController-ver2/FPGA Bitfiles/everythingdaq_FPGATarget2_FPGAESRver5_RELKYtnkXk4.lvbitx"

    #### Give the different pulse parameters
    aomvolt=config['aomvolt'] # !!! In VOLTS, ONLY 0-1V
    pulsenum=config['pulsenum'] 
    countt=config['count_t'] # Durations in ns ONLY
    separationt=config['separation_t'] # Durations in ns ONLY 
    addlt=config['addl_t'] # Durations in ns ONLY 
    waitt=config['wait_t'] # Durations in ns ONLY
    freq = config['freq'] # Hz
    mw_power=config['mw_power'] # dBm

    # #### Connect to Pulsestreamer and setup the pulses
    pulsestreamer_ip='169.254.8.2'
    ps=pulsestreamer.PulseStreamer(pulsestreamer_ip)
    ps.reset()
    aompatt=[(96,aomvolt)] # AOM analog modulation voltage, use to control 532nm laser power
    ch0patt=[(96,1)] # Keep the laser always ON for CW ODMR
    ch1patt=[(waitt,1), (countt,1), (addlt,1),(waitt,0), (countt,0), (addlt,0),(separationt,0)]  # Microwave
    ch2patt=[(waitt,0), (countt,1), (addlt,0),(waitt,0), (countt,1), (addlt,0),(separationt,0)]  # Counting
    seq=ps.createSequence()
    seq.setAnalog(0,aompatt)
    seq.setDigital(0,ch0patt)
    seq.setDigital(1,ch1patt)
    seq.setDigital(2,ch2patt)

    if config['seqplot']: seq.plot()


    # #### Open and configure the SG386
    rm=pyvisa.ResourceManager()
    # # device_list=rm.list_resources()
    # # print(device_list)
    sg386=rm.open_resource('GPIB0::27::INSTR')
    
    sg386.write('AMPR '+str(mw_power) )
    print(sg386.query('AMPR?'))
    
    ### Connect to FPGA, stream pulse sequence, and read the FIFO Buffer    
    with nifpga.Session(bitfile=bitfile_loc, resource="RIO0") as session:
        sg386.write('FREQ '+str(freq))
        # print(sg386.query('FREQ?'))
        session.reset()
        session.run()
        target2host=session.fifos['FIFO_target2host']
        target2host.configure(pulsenum)
        target2host.start()
        ps.stream(seq,n_runs=int(pulsenum/2))
        # print(target2host.name)
        read_value = target2host.read(pulsenum,1000)
        target2host.stop()

        counts=read_value[0]
        contrast=sum(counts[0::2])/sum(counts[1::2])
    
    sg386.close()

    return contrast,sum(counts[0::2])*1e9/(len(counts)*countt),sum(counts[1::2])*1e9/(len(counts)*countt)


def confocal_ODMR_Scan(config_confocal,config_ODMR):
    controller=mdt69x.Controller()
    # x, y, z = controller.get_xyz_voltage()
    # print("X: %.4f V Y: %.4f V Z: %.4f V" % (x, y, z))
    v_min=config_confocal['v_min']
    v_max=config_confocal['v_max']
    v_step=config_confocal['v_step']
    z_vol=config_confocal['z_vol']
    t_col=config_confocal['t_col']
    aomvolt=config_confocal['aomvolt']

    n_step=int((v_max-v_min)/v_step)+1
    vol_range=np.linspace(v_min,v_max,n_step)
    confocal=np.zeros([n_step,n_step])
    ODMR_contrast=np.zeros([n_step,n_step])
    ODMR_mw=np.zeros([n_step,n_step])
    ODMR_nomw=np.zeros([n_step,n_step])
    for j in range(n_step):
        for i in range(n_step):
            controller.set_xyz_voltage(vol_range[i],vol_range[j],z_vol)
            time.sleep(10e-3)
            confocal[i][j]=getConfocalCounts(t_col, aomvolt)/t_col
            ODMR=getODMRContrast(config_ODMR)
            ODMR_contrast[i][j]=ODMR[0]
            ODMR_mw[i][j]=ODMR[1]
            ODMR_nomw[i][j]=ODMR[2]

    controller.set_xyz_voltage(0,0,z_vol)
    controller.close()
    return confocal, ODMR_contrast,ODMR_mw, ODMR_nomw


### Confocal Scan Function Definitions
def getConfocalCounts(tcol, aomvolt):
    # aomvolt=config_confocal['aomvolt'] # !!! In VOLTS, ONLY 0-1V
    pulsestreamer_ip='169.254.8.2'
    ps=pulsestreamer.PulseStreamer(pulsestreamer_ip)
    ps.reset()
    # aompatt=[(96,aomvolt)] # AOM analog modulation voltage, use to control 532nm laser power
    # ch0patt=[(96,1)] # Keep the laser always ON for confocal scan
    # seq=ps.createSequence()
    # seq.setAnalog(0,aompatt)
    # seq.setDigital(0,ch0patt)
    # ps.stream(seq,n_runs=10)
    ps.constant(([0],aomvolt,0))

    bitfile="C:/NDController-ver2/FPGA Bitfiles/everythingdaq_FPGATarget2_SimplifiedCounte_YC8FIx4q1RA.lvbitx"
    with nifpga.Session(bitfile=bitfile, resource="RIO0") as session:               
        tickreg=session.registers['Count(Ticks)']
        tickreg.write(int(tcol*4e7)) # Input after converting to number of ticks
        fifoin=session.fifos['FIFO_host2target']    
        fifoin.configure(100)
        fifoin.start()
        fifoin.write([0,1])        
        time.sleep(tcol)
        count_reg=session.registers['Counts']
        countout=count_reg.read()        
        return countout
    
def confocalScan(config_confocal):  
    controller=mdt69x.Controller()
    aomvolt=config_confocal['aomvolt'] # !!! In VOLTS, ONLY 0-1V

    v_min=config_confocal['v_min']
    v_max=config_confocal['v_max']
    v_step=config_confocal['v_step']
    z_vol=config_confocal['z_vol']
    t_col=config_confocal['t_col']
    # x, y, z = controller.get_xyz_voltage()
    # print("X: %.4f V Y: %.4f V Z: %.4f V" % (x, y, z))
    v_min=0
    n_step=int((v_max-v_min)/v_step)+1
    vol_range=np.linspace(v_min,v_max,n_step)
    confocal=np.zeros([n_step,n_step])
    for j in range(n_step):
        for i in range(n_step):
            controller.set_xyz_voltage(vol_range[i],vol_range[j],z_vol)
            # time.sleep(t_col)
            confocal[i][j]=getConfocalCounts(t_col, aomvolt)/t_col
    controller.set_xyz_voltage(0,0,z_vol)
    controller.close()
    return confocal
    
# controller.close()