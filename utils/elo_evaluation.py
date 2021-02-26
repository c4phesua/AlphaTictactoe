import random
from copy import deepcopy

from tictactoe_models.tictactoe_player import DQNPlayer
from tictactoe_models.tictactoe_v0 import Tictactoe_v0
from multiprocessing import Process, Queue
from multiprocessing.managers import ValueProxy
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.layers import Dense


class Evaluator(Process):
    def __init__(self, queue: Queue, scores_list: Queue, best_player: ValueProxy):
        super().__init__()
        self.queue = queue
        self.scores_list = scores_list
        self.best_player = best_player

    def run(self) -> None:
        player = DQNPlayer()
        optimizer = optimizers.RMSprop(learning_rate=0.00025)
        player.brain.add(Dense(128, activation='relu', input_shape=(9,)))
        player.brain.add(Dense(128, activation='relu'))
        player.brain.add(Dense(9, activation='linear'))
        player.brain.compile(optimizer=optimizer, loss=losses.mean_squared_error)
        while True:
            if not self.queue.empty():
                next_weight = self.queue.get()
                print('queue size: ', self.queue.qsize())
                if next_weight is None:
                    break
                player.brain.set_weights(next_weight)
                player.elo = 0
                self.evaluate_player(player)
                self.scores_list.put(player.get_elo())
                if player.get_elo() > self.best_player.value.get_elo():
                    self.best_player.value = deepcopy(player)

    @staticmethod
    def evaluate_player(player: DQNPlayer):
        print("--- start evaluation process ---")
        result_score = {1: 1, 0: 0.5, -1: 0}
        env_test = Tictactoe_v0()
        for ep in range(300):
            done = False
            reward = 0
            state = env_test.reset(1)
            is_first_move = True
            print('game ' + str(ep) + ' start ---------------------')
            while not done:
                if is_first_move:
                    action = random.choice([i for i in range(len(state)) if state[i] == 0])
                    is_first_move = False
                else:
                    action = player.observe(state, [i for i in range(len(state)) if state[i] == 0])
                state, reward, done, _ = env_test.step(action)
            player.update_elo(1000, result_score[reward])
