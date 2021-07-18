import random
from Tictactoe.tictactoe_player import BasePlayer


class Tictactoe_v0:
    def __init__(self, player_1: BasePlayer, player_2: BasePlayer):
        self.board = [0] * 9
        self.wining_position = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                                [0, 3, 6], [1, 4, 7], [2, 5, 8],
                                [0, 4, 8], [6, 4, 2]]
        self.current_turn = 1
        self.player_mark = 1
        self.player_1 = player_1
        self.player_2 = player_2

    def reset(self, first_player=None):
        if first_player not in [0, 1]:
            raise Exception('first_player must be a integer number 0 or 1')
        self.board = [0] * 9
        self.current_turn = 1
        self.player_mark = 1 if is_human_first else -1
        if not is_human_first:
            self.env_act()
        return self.board.copy()

    def check_win(self):
        for pst in self.wining_position:
            if str(self.board[pst[0]]) + str(self.board[pst[1]]) + str(self.board[pst[2]]) in ['111', '-1-1-1']:
                if self.current_turn == self.player_mark:
                    return 1, True
                return -1, True
        if 0 not in self.board:
            return 0, True
        return 0, False

    def env_act(self):
        action = random.choice([i for i in range(len(self.board)) if self.board[i] == 0])
        for pst in self.wining_position:
            com = str(self.board[pst[0]]) + str(self.board[pst[1]]) + str(self.board[pst[2]])
            if com.replace('0', '') == str(self.current_turn) * 2:
                if self.board[pst[0]] == 0:
                    action = pst[0]
                elif self.board[pst[1]] == 0:
                    action = pst[1]
                else:
                    action = pst[2]
        if self.board[action] != 0:
            raise Exception('Invalid action')
        self.board[action] = self.current_turn
        reward, done = self.check_win()
        self.current_turn = self.current_turn * -1
        return reward, done

    def step(self, action):
        if self.board[action] != 0:
            raise Exception('Invalid action')
        self.board[action] = self.current_turn
        reward, done = self.check_win()
        self.current_turn = self.current_turn * -1
        if done:
            return self.board.copy(), reward, done, None
        reward, done = self.env_act()
        return self.board.copy(), reward, done, None
