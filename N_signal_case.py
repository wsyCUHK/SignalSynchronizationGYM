# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 01:37:56 2020

@author: admin
"""


from SignalSynchronization import Signal
from SignalSynchronization import SignalSync_Env
from SignalSynchronization import SignalSync_process
import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

"""
#Tree structure for the example
#
#S1______S2_____S4
#    |
#    |___S3_____S5
#          |
#          |____S6
#
"""
#
rnd = random.sample(range(-50, 50), 10)
s1 = Signal(0,20 )
s2 = Signal(rnd[0],rnd[1] )
s3 = Signal(rnd[2],rnd[3] )
s4 = Signal(rnd[4],rnd[5] )
s5 = Signal(rnd[6],rnd[7] )
s6 = Signal(rnd[8],rnd[9] )
signal_list = [s1, s2, s3, s4, s5, s6]
s1.AttachChild(s2)
s1.AttachChild(s3)
s2.AttachChild(s4)
s3.AttachChild(s5)
s3.AttachChild(s6)

"""
other parameters setting
"""

time_interval = 5
Max_timeout = 100
Max_interation = Max_timeout / time_interval
noise_type = 'Normal'
noise_para = np.array([0,0.1])
epsilon = 1

"""
create environment and process
"""
signal_sync_env = SignalSync_Env(signal_list,noise_type,noise_para,epsilon)
simpy_env = simpy.Environment()
signal_sync_process = SignalSync_process(simpy_env,signal_sync_env,time_interval)


"""
run the process
"""
simpy_env.run(until = Max_timeout)


"""
plot result
"""
t = np.arange(0,Max_interation,1)

plt.plot(t,s1.value_trace,'r-',label="S1")
plt.plot(t,s2.value_trace,'b--',label="S2")
plt.plot(t,s3.value_trace,'m-.',label="S3")
plt.plot(t,s4.value_trace,'g:',label="S4")
plt.plot(t,s5.value_trace,'y-.',label="S5")
plt.plot(t,s6.value_trace,'c--',label="S6")

plt.legend(loc="upper left")
plt.ylabel('Signal value')
plt.show()