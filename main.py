from multiprocessing import Queue, Manager

from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense

from rf_model.dqn_model import DQN
from tictactoe_models.tictactoe_player import DQNPlayer
from tictactoe_models.tictactoe_v0 import Tictactoe_v0
from utils.elo_evaluation import Evaluator

if __name__ == '__main__':
    # prepare model and init neural network
    env = Tictactoe_v0()
    agent = DQN(0.7, 1, 4096, 1048576)
    op1 = optimizers.RMSprop(learning_rate=0.00025)
    agent.training_network.add(Dense(128, activation='relu', input_shape=(9,)))
    agent.training_network.add(Dense(128, activation='relu'))
    agent.training_network.add(Dense(9, activation='linear'))
    agent.training_network.compile(optimizer=op1, loss=losses.mean_squared_error)

    op2 = optimizers.RMSprop(learning_rate=0.00025)
    agent.target_network.add(Dense(128, activation='relu', input_shape=(9,)))
    agent.target_network.add(Dense(128, activation='relu'))
    agent.target_network.add(Dense(9, activation='linear'))
    agent.target_network.compile(optimizer=op2, loss=losses.mean_squared_error)
    agent.update_target_network()
    reward_records = list()
    loss_records = list()
    count = 0
    tau = 5  # number for delay update weight from training network to target network
    omega = 5  # system will execute evaluation process after every [omega] episodes
    record = 0
    weight_queue = Queue()
    scores = Queue()
    best_player = Manager().Value(DQNPlayer, None)
    evaluator = Evaluator(weight_queue, scores, best_player)
    evaluator.start()

    # start training process
    for ep in range(100):
        state = env.reset(1)
        done = False
        print(ep, '------------------', 'current epsilon: ', agent.epsilon_greedy.epsilon)
        while not done:
            action = agent.observe_on_training(state, [i for i in range(len(state)) if state[i] == 0])
            state, reward, done, _ = env.step(action)
            print(state, done)
            record += reward
            print(ep, '-----------------------------------', reward)
            agent.take_reward(reward, state, done)
            hist = agent.train_network(64, 64, 1, 2, cer_mode=True)
            loss_records.append(hist)
            count += 1
            if count % tau == 0:
                agent.update_target_network()
        if ep % omega == 0:
            weight_queue.put(agent.target_network.get_weights())
        reward_records.append(record)
        agent.epsilon_greedy.decay(0.99999, 0.1)
    weight_queue.put(None)
    evaluator.join()
    # show scores
    while not scores.empty():
        print(scores.get())
