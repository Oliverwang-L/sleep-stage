import sys
sys.path.append('../')
sys.path.append('../../')

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
gpu_id = 1

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('allow_growth', False)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)

s.register('lr', 0.0002, 0.0008, hp_type=list)
s.register('batch_size', 32, 64, 128, hp_type=list)
s.register('archi_string', '3-3-3', '5-5-5')
s.register('aug_config', 'flip:True;False|rotate',
           'flip:True;False')

s.configure_engine(strategy='skopt', criterion='Best Accuracy')
s.run(rehearsal=False)