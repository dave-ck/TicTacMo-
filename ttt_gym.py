# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/simple_dqn_torch.py
import ttt_env
from phils_dq import DeepQNetwork, Agent
from phils_utils import plotLearning
import numpy as np
from gym import wrappers
from test import check_nan

def play_brains(n, k, q=2, gamma=0.99, epsilon=1.0, alpha=0.003, num_games=1000):
    # derive batch_size, n_actions, input_dims from n, k
    n_actions = n ** k
    input_dims = [n ** k]
    env = ttt_env.nkq_game(n=n, k=k)
    # todo: whomst'd've batch_size?
    brains = [Agent(gamma=gamma, epsilon=epsilon, batch_size=64, n_actions=n_actions,
                    input_dims=input_dims, alpha=alpha) for _ in range(q)]
    print(brains)
    scores_history = {i: [] for i in range(q)}
    eps_history = []
    game_scores = {i: 0 for i in range(q)}
    for i in range(num_games):
        if i % 10 == 0 and i > 0:
            logfile = open('logfile.txt', 'a')
            avg_scores = ['%.3f' % np.mean(scores_history[player][max(0, i - 10):i + 1]) for player in range(q)]
            scores = [game_scores[player] for player in range(q)]
            print('episode: ', i, 'scores: ', scores,
                  ' average scores', avg_scores,
                  'epsilon %.3f' % brains[0].EPSILON)  # brains all have same epsilon
            logfile.write('episode: ' + str(i) + 'scores: ' + str(scores) +
                          ' average scores' + str(avg_scores) +
                          'epsilon %.3f' % brains[0].EPSILON)
            logfile.close()
        else:
            # scores = [np.mean(game_scores[player]) for player in range(q)]
            # print('episode: ', i, 'scores: ', scores)
            pass
        display_board = False
        if i % 100 == 0 and i > 0:
            display_board = True
        eps_history.append(brains[0].EPSILON)
        done = False
        current_observation = env.reset()
        last_action = {i: None for i in range(q)}
        last_observation = {i: None for i in range(q)}
        while not done:
            for player in range(q):
                if last_action[player] is not None:  # if player has played, back-propagate outcome of last action
                    brains[player].storeTransition(last_observation[player],
                                                   last_action[player],
                                                   0.,  # reward is *always* 0 unless game ended
                                                   current_observation,
                                                   done)
                    check_nan(last_observation[player])
                    # check_nan(last_action[player])
                    check_nan(current_observation)
                    brains[player].learn()
                action = brains[player].chooseAction(current_observation)  # choose action from current knowledge
                last_action[player] = action
                current_observation, reward, done, info = env.step(action,
                                                                   display_board)  # make move in the environment
                last_observation[player] = current_observation
                if done:
                    for player_ in range(q):  # back propagate players' wins/losses/draws
                        if last_action[player_] is not None:  # only if player has played
                            # no need i.e. when player=4, n=2 - never had a chance to affect game
                            brains[player_].storeTransition(last_observation[player_],
                                                            last_action[player_],
                                                            reward[player_],
                                                            current_observation,
                                                            done)
                            scores_history[player_].append(reward[player_])
                            game_scores[player_] = reward[player_]
                    break
    for player in range(q):
        x = [i + 1 for i in range(len(scores_history[player]))]
        filename = "tmp/{n}n{k}k{q}q_{games}games_player{player}_" \
                   "gamma{gamma}alpha{alpha}mem{fc1}-{fc2}.png".format(n=n,
                                                                       k=k,
                                                                       q=q,
                                                                       games=num_games,
                                                                       player=player,
                                                                       alpha=brains[player].ALPHA,
                                                                       gamma=brains[player].GAMMA,
                                                                       fc1=brains[player].Q_eval.fc1_dims,
                                                                       fc2=brains[player].Q_eval.fc2_dims)
        plotLearning(x, scores_history[player], eps_history, filename)


def old_gym():
    env = ttt_env.nkq_game(n=3, k=2)
    brain = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=9,
                  input_dims=[9], alpha=0.003)
    scores = []
    eps_history = []
    num_games = 1000
    score = 0
    # uncomment the line below to record every episode.
    # env = wrappers.Monitor(env, "tmp/space-invaders-1",
    # video_callable=lambda episode_id: True, force=True)
    for i in range(num_games):
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0, i - 10):(i + 1)])
            print('episode: ', i, 'score: ', score,
                  ' average score %.3f' % avg_score,
                  'epsilon %.3f' % brain.EPSILON)
        else:
            print('episode: ', i, 'score: ', score)
        display_board = False
        if i % 100 == 0 and i > 0:
            display_board = True
        eps_history.append(brain.EPSILON)
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = brain.chooseAction(observation)
            observation_, reward, done, info = env.step(action, display_board)
            score += reward
            brain.storeTransition(observation, action, reward, observation_,
                                  done)
            observation = observation_
            brain.learn()

        scores.append(score)

    x = [i + 1 for i in range(num_games)]
    filename = str(env.n) + "n" + str(env.k) + "k" + str(num_games) + 'Games' + 'Gamma' + str(brain.GAMMA) + \
               'Alpha' + str(brain.ALPHA) + 'Memory' + \
               str(brain.Q_eval.fc1_dims) + '-' + str(brain.Q_eval.fc2_dims) + '.png'
    plotLearning(x, scores, eps_history, filename)


# todo: what does alpha do?
# todo: what does gamma do?
# todo: make fc1 and fc2 dims passable as parameters. experiment with having an fc3
play_brains(3, 2, num_games=5000, epsilon=1.0, gamma=.99, alpha=0.003)
