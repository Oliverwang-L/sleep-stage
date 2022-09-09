import os, sys

from pictor import Pictor
from pictor_sleep.monitor import Monitor
from pictor_sleep.sleep_data import SleepData



# Load data
# ...
# sleep_data_list = [SleepData(), SleepData(), ]
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ucddb')
sleep_data_list = SleepData.read_ucddb(path)

# Initiate a pictor
p = Pictor(title='Sleep Monitor', figure_size=(15, 9))

# Set plotter
m = Monitor()
p.add_plotter(m)

# Set objects
p.objects = sleep_data_list

# Begin main loop
p.show()







