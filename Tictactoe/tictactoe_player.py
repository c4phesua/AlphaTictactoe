from math import ceil, floor

from DQN.dqn_model import DQN


class BasePlayer:
    def __init__(self, elo):
        self.elo = elo

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

    def make_move(self, state, mode, action_space: list = None):
        raise NotImplementedError

    def get_elo(self):
        return int(self.elo)


class DQNPlayer(BasePlayer, DQN):
    def __init__(self, elo, discount_factor: float, epsilon: float, e_min: int, e_max: int):
        BasePlayer.__init__(self, elo)
        DQN.__init__(self, discount_factor, epsilon, e_min, e_max)

    def make_move(self, state, mode: str, action_space: list = None):
        if mode.lower() not in ['training', 'evaluate']:
            raise Exception('mode must be training or evaluate')
        if mode.lower() == 'training':
            return self.observe_on_training(state, action_space)
        return self.observe(state, action_space)
