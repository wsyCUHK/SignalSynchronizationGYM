import SignalSynchronization
import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

"""
Two signal example
S1 is required to synchronized to S2 
Therefore, S2 is the parent signal, S1 is the child signal

"""



"""
signal setting

"""

s2 = SignalSynchronization.Signal(0,10)
s1 = SignalSynchronization.Signal( random.sample(range(-50, 50), 1), random.sample(range(0, 20), 1))
signal_list = [s1,s2]
s2.AttachChild(s1)

"""
other parameters setting
"""

time_interval = 5
Max_timeout = 50
Max_interation = Max_timeout / time_interval
noise_type = 'Normal'
noise_para = np.array([0,0.1])
epsilon = 1

"""
create environment and process
"""
signal_sync_env = SignalSynchronization.SignalSync_Env(signal_list,noise_type,noise_para,epsilon)
simpy_env = simpy.Environment()
signal_sync_process = SignalSynchronization.SignalSync_process(simpy_env,signal_sync_env,time_interval)


"""
run the process
"""
simpy_env.run(until = Max_timeout)


"""
plot result and print total reward
"""


print('Total reward %d' %signal_sync_process.total_reward[0])

RMSE = np.sqrt(((s1.value_trace - s2.value_trace) ** 2).mean())
print('RMSE %f' %RMSE)


t = np.arange(0,Max_interation,1)
plt.plot(s1.value_trace,'b-',label="S1")
plt.plot(s2.value_trace,'r--',label="S2")
plt.legend(loc="upper left")
plt.ylabel('Signal value')
plt.show()
