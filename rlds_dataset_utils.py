import functools
import rlds
import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Any, Union, Callable

import numpy as np
import tensorflow_datasets as tfds
import tree
from rlds_dataset_classes import RLDSSpec, TrajectoryTransformBuilder


def n_step_pattern_builder(n: int) -> Any:
  """Creates trajectory of length `n` from all fields of a `ref_step`."""

  def transform_fn(ref_step):
    traj = {}
    for key in ref_step:
      if isinstance(ref_step[key], dict):
        transformed_entry = tree.map_structure(lambda ref_node: ref_node[-n:],
                                               ref_step[key])
        traj[key] = transformed_entry
      else:
        traj[key] = ref_step[key][-n:]

    return traj

  return transform_fn

def pad_initial_zero_steps(
    steps: tf.data.Dataset, num_zero_step: int
) -> tf.data.Dataset:
  zero_steps = steps.take(1)
  zero_steps = zero_steps.map(lambda x: tf.nest.map_structure(tf.zeros_like, x),
                              num_parallel_calls=tf.data.AUTOTUNE)
  zero_steps = zero_steps.repeat(num_zero_step)
  return rlds.transformations.concatenate(zero_steps, steps)

def pad_initial_zero_episode(episode: tf.data.Dataset, num_zero_step: int) -> tf.data.Dataset:
  episode[rlds.STEPS] = pad_initial_zero_steps(episode[rlds.STEPS], num_zero_step)
  return episode

def terminate_bool_to_act(terminate_episode: tf.Tensor) -> tf.Tensor:
  return tf.cond(
      terminate_episode == tf.constant(1.0),
      lambda: tf.constant([1, 0, 0], dtype=tf.int32),
      lambda: tf.constant([0, 1, 0], dtype=tf.int32),
  )

def get_trajectory_dataset(builder_dir: str, step_map_fn, trajectory_length: int, split='train'):
  dataset_builder = tfds.builder_from_directory(builder_dir=builder_dir)
  
  dataset_builder_episodic_dataset = dataset_builder.as_dataset(split=split)
  # We need pad_initial_zero_episode because reverb.PatternDataset will skip
  # constructing trajectories where the first trajectory_length - 1 steps are
  # the final step in a trajectory. As such, without padding, the policies will
  # not be trained to predict the actions in the first trajectory_length - 1
  # steps.
  # We are padding with num_zero_step=trajectory_length-1 steps.
  dataset_builder_episodic_dataset = dataset_builder_episodic_dataset.map(
      functools.partial(pad_initial_zero_episode, num_zero_step=trajectory_length-1), num_parallel_calls=tf.data.AUTOTUNE)


  print(next(iter(dataset_builder_episodic_dataset))['steps'])
  
  rlds_spec = RLDSSpec(
      observation_info=dataset_builder.info.features[rlds.STEPS][rlds.OBSERVATION],
      action_info=dataset_builder.info.features[rlds.STEPS][rlds.ACTION],
  )

  trajectory_transform = TrajectoryTransformBuilder(rlds_spec,
                                                    step_map_fn=step_map_fn,
                                                    pattern_fn=n_step_pattern_builder(trajectory_length)).build(validate_expected_tensor_spec=False)

  trajectory_dataset = trajectory_transform.transform_episodic_rlds_dataset(dataset_builder_episodic_dataset)

  return trajectory_dataset

StepFnMapType = Callable[[rlds.Step, rlds.Step], None]

def step_map_fn(step, map_observation: StepFnMapType, map_action: StepFnMapType):
  transformed_step = {}
  transformed_step[rlds.IS_FIRST] = step[rlds.IS_FIRST]
  transformed_step[rlds.IS_LAST] = step[rlds.IS_LAST]
  transformed_step[rlds.IS_TERMINAL] = step[rlds.IS_TERMINAL]

  transformed_step[rlds.OBSERVATION] = {}
  transformed_step[rlds.ACTION] = {
    'gripper_closedness_action': tf.zeros(1, dtype=tf.float32),
    'rotation_delta': tf.zeros(3, dtype=tf.float32),
    'terminate_episode': tf.zeros(3, dtype=tf.int32),
    'world_vector': tf.zeros(3, dtype=tf.float32)
  }

  map_observation(transformed_step, step)
  map_action(transformed_step, step)

  return transformed_step

def viola_map_action(to_step: rlds.Step, from_step: rlds.Step):
  """Maps dataset action to action expected by the model."""

  # The world vector as existed in the dataset on disk ranges from -1.0 to 1.0
  # We scale by 1.75 so that the action better spans the limit of the
  # world_vector action, from -2.0 to 2.0.
  to_step[rlds.ACTION]['world_vector'] = from_step[rlds.ACTION]['world_vector'] * 1.75
  to_step[rlds.ACTION]['terminate_episode'] = terminate_bool_to_act(
      from_step[rlds.ACTION]['terminate_episode']
  )

  # Similarly, the rotation_delta in the dataset on disk ranges from -0.4 to 0.4
  # We scale by 3.0 so that the rotation_delta almost spans the limit of
  # rotation_delta, from -pi/2 to pi/2.
  to_step[rlds.ACTION]['rotation_delta'] = (
      from_step[rlds.ACTION]['rotation_delta'] * 3.0
  )
  
  

  gripper_closedness_action = from_step[rlds.ACTION]['gripper_closedness_action']

  # There can be 0.0 values because of zero padding
  possible_values = tf.constant([-1.0, 1.0, 0.0], dtype=tf.float32)
  eq = tf.equal(possible_values, gripper_closedness_action)

  # Assert that gripper_closedness_action is one of possible_values
  assert_op = tf.Assert(tf.reduce_any(eq), [gripper_closedness_action])

  with tf.control_dependencies([assert_op]):
    gripper_closedness_action = tf.expand_dims(
        gripper_closedness_action, axis=-1
    )
    to_step[rlds.ACTION]['gripper_closedness_action'] = gripper_closedness_action

def map_observation(
    to_step: rlds.Step,
    from_step: rlds.Step,
    from_image_feature_names: tuple[str, ...] = ('image',),
    to_image_feature_names: tuple[str, ...] = ('image',),
    resize: bool = True,
    target_width=320,
    target_height=256,
    ) -> None:
  """Map observation to model observation spec."""

  to_step[rlds.OBSERVATION]['natural_language_embedding'] = from_step[
      rlds.OBSERVATION
  ]['natural_language_embedding']

  for from_feature_name, to_feature_name in zip(
      from_image_feature_names, to_image_feature_names
  ):
    if resize:
      to_step['observation'][to_feature_name] = resize_to_resolution(
          from_step['observation'][from_feature_name],
          to_numpy=False,
          target_width=target_width,
          target_height=target_height,
      )

def resize_to_resolution(
    image: Union[tf.Tensor, np.ndarray],
    target_width: int = 320,
    target_height: int = 256,
    to_numpy: bool = True,
    ) -> Union[tf.Tensor, np.ndarray]:
  """Resizes image and casts to uint8."""
  image = tf.image.resize_with_pad(
      image,
      target_width=target_width,
      target_height=target_height,
  )
  image = tf.cast(image, tf.uint8)
  if to_numpy:
    image = image.numpy()
  return image

viola_map_observation = functools.partial(
    map_observation,
    from_image_feature_names = ('agentview_rgb',),
    to_image_feature_names = ('image',),
)