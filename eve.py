# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/simple_dqn_torch.py
from obsolete import ttt_env
from dq import Agent
from obsolete.phils_utils import plotLearning
import numpy as np
from board import Board


def play(num_games, n, k, q):
    brains = {
    i: Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=n ** k, num_pos=n ** k, alpha=0.003, eps_end=0.005,
             eps_dec=0.9999, q=q) for i in range(1, q + 1)}
    scores = {i: [] for i in range(1, q + 1)}
    eps_history = []
    board = Board.blank_board(n, k, q)
    for i in range(num_games):
        if i % 40 == 0 and i > 0:
            out = 'episode: %d' % i
            for player in range(1, q+1):
                avg_score = np.mean(scores[player][max(0, i - 40):(i + 1)])
                out += '; average score[%d]: %.3f' % (player, avg_score)
            out += '; epsilon: %.3f' % brains[1].EPSILON
            print(out)
        eps_history.append(brains[1].EPSILON)
        done = False
        board.reset()
        action0, action1 = -1, -1
        # set actions to values >= 0 as each player gets their first move; can add a test for large 1 inside while-loop
        last_observation0, last_observation1 = None, None
        win0, win1, draw = False, False, False
        while not done:
            last_observation0 = board.to_linear_array()
            # print(last_observation0)
            action0 = brains[1].chooseAction(last_observation0)
            board.move(action0)
            win0 = board.win() == 1  # bool on whether win for X
            draw = board.draw()
            if win0:
                brains[1].storeTransition(last_observation0, action0, 5, board.to_linear_array(), True)
                brains[2].storeTransition(last_observation1, action1, -100, board.to_linear_array(), True)
                break
            if draw:
                brains[1].storeTransition(last_observation1, action0, 1, board.to_linear_array(), True)
                brains[2].storeTransition(last_observation0, action1, 1, board.to_linear_array(), True)
                break
            if action1 != -1:
                brains[2].storeTransition(last_observation1, action1, 0, board.to_linear_array(), False)
                brains[2].learn()
            last_observation1 = board.to_linear_array()
            # print(last_observation1)
            action1 = brains[2].chooseAction(last_observation1)
            board.move(action1)
            win1 = board.win() == 2  # bool on whether win for O
            draw = board.draw()
            if win1:
                brains[1].storeTransition(last_observation1, action1, -100, board.to_linear_array(), True)
                brains[2].storeTransition(last_observation0, action0, 5, board.to_linear_array(), True)
                break
            if draw:
                brains[1].storeTransition(last_observation1, action0, 1, board.to_linear_array(), True)
                brains[2].storeTransition(last_observation0, action1, 1, board.to_linear_array(), True)
                break
            brains[1].storeTransition(last_observation0, action0, 0, board.to_linear_array(), False)
            brains[1].learn()
        if win0:
            score0 = 1
            score1 = -1
        elif win1:
            score0 = -1
            score1 = 1
        elif draw:
            score0 = 0
            score1 = 0
        else:
            raise Exception("Should not be reachable; exited while-loop without win0, win1, or draw.")
        scores[1].append(score0)
        scores[2].append(score1)
    x = [i + 1 for i in range(num_games)]
    # filename = "tmp/" + str(env.n) + "n" + str(env.k) + "k" + str(num_games) + 'Games' + 'Gamma' + str(brain.GAMMA) + \
    #            'Alpha' + str(brain.ALPHA) + 'Memory' + \
    #            str(brain.Q_eval.fc1_dims) + '-' + str(brain.Q_eval.fc2_dims) + '.png'
    # todo: update when player turn implemented
    # modelname = "models/p{p}_q{q}_n{n}_k{k}_games{games}_fc1{fc1}_fc2{fc2}_alpha{alpha}_.pth" \
    #     .format(p=0, q=2, n=env.n, k=env.k, games=num_games, fc1=brain.Q_eval.fc1_dims, fc2=brain.Q_eval.fc2_dims,
    #             alpha=brain.Q_eval.alpha)
    plotLearning(x, brains[1], eps_history, "./tmp/{ng}brain0_n{n}_k{k}.png".format(ng=num_games, n=n, k=k))
    plotLearning(x, brains[2], eps_history, "./tmp/{ng}brain1_n{n}_k{k}.png".format(ng=num_games, n=n, k=k))
    # brain.save_model(modelname)


play(1000, 3, 2, 2)
