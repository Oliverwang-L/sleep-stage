import sys
sys.path.append('../')

from tframe.utils.script_helper import Helper
s = Helper()

from fm_core import th
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
pass

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
gpu_id = 0

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('allow_growth', False)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)
s.register('patience', 10)

s.register('lr', 0.0001,0.0003, 0.001, 0.01)
s.register('batch_size', 32, 64, 128)
s.register('archi_string', '64-p-32-p-32-p-16-p-16-p-8', '32-p-16-p-8', '5-p-10-p-15-p-20-p-25')

s.configure_engine(times=10)
s.run(rehearsal=False)