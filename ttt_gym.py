# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/simple_dqn_torch.py
import ttt_env
from phils_dq import DeepQNetwork, Agent
from phils_utils import plotLearning
import numpy as np
from gym import wrappers


def play(num_games):
    env = ttt_env.nkq_game(n=5, k=2)
    brain = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=25,
                  input_dims=[25], alpha=0.003, eps_end=0.001, eps_dec=0.999)

    scores = []
    eps_history = []
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
    filename = "tmp/" + str(env.n) + "n" + str(env.k) + "k" + str(num_games) + 'Games' + 'Gamma' + str(brain.GAMMA) + \
               'Alpha' + str(brain.ALPHA) + 'Memory' + \
               str(brain.Q_eval.fc1_dims) + '-' + str(brain.Q_eval.fc2_dims) + '.png'
    # todo: update when player turn implemented
    modelname = "models/p{p}_q{q}_n{n}_k{k}_games{games}_fc1{fc1}_fc2{fc2}_alpha{alpha}_.pth" \
        .format(p=0, q=2, n=env.n, k=env.k, games=num_games, fc1=brain.Q_eval.fc1_dims, fc2=brain.Q_eval.fc2_dims,
                alpha=brain.Q_eval.alpha)
    plotLearning(x, scores, eps_history, filename)
    brain.save_model(modelname)


play(7504)
