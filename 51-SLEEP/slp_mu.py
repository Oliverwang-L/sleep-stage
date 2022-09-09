from tframe import Classifier
from tframe import mu

from slp_core import th



def get_container(flatten=False):
  model = Classifier(mark=th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  if th.centralize_data: model.add(mu.Normalize(mu=th.data_mean))
  if flatten: model.add(mu.Flatten())

  # if 'expand' in th.developer_code: model.add(mu.Reshape(th.input_shape + [1]))

  return model


def finalize(model):
  assert isinstance(model, Classifier)
  model.add(mu.Dense(th.output_dim, activation='softmax'))

  # Build model
  model.build(batch_metric=['accuracy'])
  return model

