from typing import Any

from tf_agents.typing import types

from Connect4 import Connect4
import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment, tf_environment
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts


class Connect4PyEnv(py_environment.PyEnvironment):
    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self) -> Any:
        return self.connect4.arr

    def set_state(self, state: Any) -> None:
        pass

    def __init__(self, connect4: Connect4, color):
        super(Connect4PyEnv, self).__init__()
        self.connect4 = connect4
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int8, minimum=0, maximum=6, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(Connect4.ROWS, Connect4.COLS),
                                                             dtype=np.int8, name='observation')
        self.color = color

    @property
    def _episode_ended(self):
        return self.connect4.gameOver

    def _reset(self):
        self.connect4 = Connect4()
        return ts.restart(self.connect4.arr)

    def _step(self, action):
        if self._episode_ended:
            return self._reset()
        if action not in self.connect4.legalMoves:
            # The agent is trying to make an illegal move (like inserting into column that is full)
            breakpoint()
            if self.connect4.turn == Connect4.COLOR_1:
                self.connect4.winner = Connect4.COLOR_2
            else:
                self.connect4.winner = Connect4.COLOR_1
        self.connect4.takeTurn(action)
        if self._episode_ended:
            reward = 100 if self.connect4 == self.connect4.winner else -100
            return ts.termination(self.connect4.arr, reward)
        else:
            return ts.transition(self.connect4.arr, reward=0, discount=1)