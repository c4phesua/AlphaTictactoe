from math import ceil, floor

from tensorflow.keras.models import Sequential
import numpy as np


class DQNPlayer:
    def __init__(self, elo: int = 0):
        self.elo = elo
        self.brain = Sequential()

    def get_expected_value(self, opponent_elo: int):
        return 1 / (1 + 10 ** ((opponent_elo - self.elo) / 400))

    def update_elo(self, opponent_elo: int, score: float):
        expect_value = self.get_expected_value(opponent_elo)
        point = 40 * (score - expect_value)
        if point > 0:
            point = ceil(point)
        else:
            point = floor(point)
        self.elo += point
        self.elo = max(0, self.elo)

    def observe(self, state, action_space: list = None):
        q_value = self.brain.predict(np.array([state])).ravel()
        if action_space is not None:
            return max([[q_value[a], a] for a in action_space], key=lambda x: x[0])[1]
        return np.argmax(q_value)

    def get_elo(self):
        return int(self.elo)
