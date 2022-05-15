from Connect4 import Connect4
import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment, tf_environment
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts


# from enum import Enum
# class States(Enum):
#     FIRST = ts.StepType.FIRST
#     MID = ts.StepType.MID
#     LAST = ts.StepType.LAST
class Connect4Env(tf_environment.TFEnvironment):

    def __init__(self, initial_state=Connect4().arr, scope='Connect4 Environment'):
        self._initial_state = initial_state
        self._scope = scope
        observation_spec = array_spec.BoundedArraySpec(shape=(Connect4.ROWS, Connect4.COLS),
                                                       dtype=np.int8, name='observation')
        action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int8, minimum=0, maximum=6, name='action')
        time_step_spec = ts.time_step_spec(observation_spec)
        super(Connect4Env, self).__init__(time_step_spec, action_spec)
        self.steps = common.create_variable('steps', 0)
        self.episodes = common.create_variable('episodes', 0)
        self.resets = common.create_variable('resets', 0)
        self._state = common.create_variable('state', initial_state)

    def _current_time_step(self):
        def start():
            return (tf.constant(ts.StepType.FIRST, name='Start State'), tf.concat(0, name='reward'),
                    tf.concat(1, name='discount'))

        def mid():
            return (tf.constant(ts.StepType.MID, name='Mid State'), tf.concat(0, name='reward'),
                    tf.concat(1, name='discount'))

        def last():
            return (tf.constant(ts.StepType.LAST, name='End State'), tf.concat(1, name='reward'),
                    tf.concat(0, name='discount'))

        state_value = tf.math.mod(self._state.value(), 3)
        stepType, reward, discount = tf.case([
            (tf.equal(state_value, ts.StepType.FIRST), start),
            (tf.equal(state_value, ts.StepType.MID), mid),
            (tf.equal(state_value, ts.StepType.LAST), last),
        ], exclusive=True, strict=True)
        return ts.TimeStep(stepType, reward, discount, state_value)

    def _reset(self):
        self.connect4.reset()
        increase_resets = self.resets.assign_add(1)
        with tf.control_dependencies([increase_resets]):
            reset_op = self._state.assign(self._initial_state)
        with tf.control_dependencies([reset_op]):
            time_step = self.current_time_step()
        return time_step

    def _step(self, action):
        action = tf.convert_to_tensor(value=action)

        with tf.control_dependencies(tf.nest.flatten(action)):
            stateAdd = self._state.assign_add(1)
        with tf.control_dependencies([stateAdd]):
            stateValue = self._state.value()
            increase_steps = tf.cond(
                pred=tf.equal(tf.math.mod(stateValue, 3), ts.StepType.FIRST),
                true_fn=self.steps.value,
                false_fn=lambda: self.steps.assign_add(1)
            )
            increase_episodes = tf.cond








if __name__ == '__main__':
    breakpoint()
