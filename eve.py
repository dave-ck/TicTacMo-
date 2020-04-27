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
            for player in range(1, q + 1):
                avg_score = np.mean(scores[player][max(0, i - 40):(i + 1)])
                out += '; average score[%d]: %.3f' % (player, avg_score)
            out += '; epsilon: %.3f' % brains[1].EPSILON
            print(out)
        # else:
        #     print("Game %d; Epsilon @ %.5f" % (i, brains[1].EPSILON))
        eps_history.append(brains[1].EPSILON)
        board.reset()
        # set actions to values >= 0 as each player gets their first move; can add a test for large 1 inside while-loop
        last_seen = {i: board.to_linear_array() for i in range(1, q + 1)}
        last_move = {i: -1 for i in range(1, q + 1)}
        winner = board.win()
        while not winner:  # during play
            player = (board.turn % board.q) + 1
            current_obs = board.to_linear_array()
            if last_move[player] != -1:  # if player has moved
                brains[player].storeTransition(last_seen[player], last_move[player],
                                               board.reward(player, offense_scaling=0.1, defense_scaling=1),
                                               current_obs, False)
                brains[player].learn()
            action = brains[player].chooseAction(current_obs)
            last_seen[player] = current_obs
            last_move[player] = action
            board.move(action)
            if i > 0 and i % 240 == 0:
                print("After move %d:" % board.turn)
                board.cli()
                print("\n\n")
            winner = board.win()
        if i > 0 and i % 240 == 0:
            print("Winner: %d" % winner)
        for player in range(1, q + 1):  # after the game has ended, all players learn
            current_obs = board.to_linear_array()
            if last_move[player] != -1:  # if player has moved
                brains[player].storeTransition(last_seen[player], last_move[player], board.reward(player), current_obs,
                                               True)
                brains[player].learn()
            scores[player].append(board.reward(player))
    x = [i + 1 for i in range(num_games)]
    # filename = "tmp/" + str(env.n) + "n" + str(env.k) + "k" + str(num_games) + 'Games' + 'Gamma' + str(brain.GAMMA) + \
    #            'Alpha' + str(brain.ALPHA) + 'Memory' + \
    #            str(brain.Q_eval.fc1_dims) + '-' + str(brain.Q_eval.fc2_dims) + '.png'
    # todo: update when player turn implemented
    for i in range(1, q + 1):
        brains[i].save_model("models/%dgames_%dn_%dk_%dq_player%d.pth" % (num_games, n, k, q, i))
        plotLearning(x, scores[i], eps_history,
                     "./tmp/{ng}brain{player}_n{n}_k{k}.png".format(player=i, ng=num_games, n=n, k=k))


def play_trainee(num_games, n, k, q, trainee_number):
    brain = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=n ** k, num_pos=n ** k, alpha=0.003, eps_end=0.005,
                  eps_dec=0.9999, q=q)
    scores = []
    eps_history = []
    board = Board.blank_board(n, k, q)
    for i in range(num_games):
        if i % 40 == 0 and i > 0:
            out = 'episode: %d' % i
            avg_score = np.mean(scores[max(0, i - 40):(i + 1)])
            out += '; average score[%d]: %.3f' % (trainee_number, avg_score)
            out += '; epsilon: %.3f' % brain.EPSILON
            print(out)
        # else:
        #     print("Game %d; Epsilon @ %.5f" % (i, brains[1].EPSILON))
        eps_history.append(brain.EPSILON)
        board.reset()
        # set actions to values >= 0 as each player gets their first move; can add a test for large 1 inside while-loop
        last_seen = board.to_linear_array()
        last_move = -1
        winner = board.win()
        opponents = {i: np.random.choice(['greedy', 'random'] + ['rl'] * 6) for i in range(1, q + 1)}
        # find opponents; 3/4 chance to be vs RL algorithm
        while not winner:  # during play
            player = (board.turn % board.q) + 1
            if player == trainee_number:
                current_obs = board.to_linear_array()
                if last_move != -1:  # if player has moved in the past
                    brain.storeTransition(last_seen, last_move,
                                          board.reward(player, offense_scaling=0.1, defense_scaling=1),
                                          current_obs, False)
                    brain.learn()
                action = brain.chooseAction(current_obs)
                last_seen = current_obs
                last_move = action
                board.move(action)
                if i > 0 and i % 240 == 0:
                    print("After move %d:" % board.turn)
                    board.cli()
                    print("\n\n")
                winner = board.win()
            else:
                if opponents[player] == 'greedy':
                    board.greedy_move()
                    winner = board.win()
                elif opponents[player] == 'random':
                    board.rand_move()
                    winner = board.win()
                else:
                    board.rl_move()
                    winner = board.win()
        if i > 0 and i % 240 == 0:
            print("Winner: %d" % winner)
        current_obs = board.to_linear_array()
        if last_move != -1:  # if player has moved
            brain.storeTransition(last_seen, last_move, board.reward(player), current_obs,
                                  True)
            brain.learn()
            scores.append(board.reward(player))
    x = [i + 1 for i in range(num_games)]
    plotLearning(x, scores, eps_history, "./tmp/{ng}games_player{p}_of{q}_n{n}_k{k}.png".format(q=q, p=trainee_number,
                                                                                                ng=num_games, n=n, k=k))
    brain.save_model("models/%dgames_%dn_%dk_%dq_player%d.pth" % (num_games, n, k, q, trainee_number))

for iter in range(50):
    for game in [(3, 2, 2),  # classic
                 (3, 2, 3),  # 3-player 3^2
                 (4, 3, 2),  # Qubic
                 (4, 3, 3)]:  # 3-player Qubic
        for player in [1, 2, 3]:
            if game[2] <= player:
                play_trainee(1025+iter, *game, player)

for iter in range(50):
    for game in [(3, 2, 2),  # classic
                 (3, 2, 3),  # 3-player 3^2
                 (4, 3, 2),  # Qubic
                 (4, 3, 3)]:  # 3-player Qubic
        for player in [1, 2, 3]:
            if game[2] <= player:
                play_trainee(10024+iter, *game, player)
