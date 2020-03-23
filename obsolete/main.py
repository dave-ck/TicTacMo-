from obsolete.board_naive import Board
from obsolete.game import Game
from obsolete.player import Randy
from obsolete.phils_dq import TF_Player, Agent
from obsolete.phils_utils import plotLearning


def evaluate_game(k, n, q):
    tf = TF_Player(path="models/p0_q2_n3_k2_games200_fc11024_fc21024_alpha0.003_.pth")  # todo ensure implements Player
    randy = Randy(k, n, q)
    game = Game(k, n, q, [randy, None])  # randy gets first 2 moves, then computer gets a move
    game.play()


def teach_tf(k, n, q, num_games, gamma, epsilon, batch_size, alpha, eps_end, eps_dec, meta=True, write_weights=True):
    brain0 = Agent(gamma=gamma, epsilon=epsilon, batch_size=batch_size, n_actions=n ** k,
                   input_dims=[n ** k], alpha=alpha, eps_end=eps_end, eps_dec=eps_dec)
    board = Board.blank_board(n, k)
    if meta:
        # assign display_board
        scores0 = []
        eps_history = []
        score0 = 0
    for i in range(num_games):
        if meta:
            pass  # print stuff to stdout every 10 episodes, every episode, whatever
            # do some other stuff like updating eps_history
        board.reset()
        done = False
        score0 = 0
        while not done:
            observation0 = board.to_linear_array()
            action0 = brain0.chooseAction(observation0)
            board.move(action0)
            # other players do their moves
            board.rand_move()

            # after all moves
            done = board.win() or board.draw()
            reward0 = board.reward(0)  # todo: update - right now AI goes 1st
            observation0_ = board.to_linear_array()  # result of taking action0 on observation0, after moves by others
            brain0.storeTransition(observation0, action0, reward0, observation0_, done)
            if meta:
                score0 += reward0
                scores0.append(score0)
    if meta:
        x = [i + 1 for i in range(num_games)]
        filename = "tmp/" + str(n) + "n" + str(k) + "k" + str(num_games) + 'Games' + 'Gamma' + str(gamma) + \
                   'Alpha' + str(alpha) + 'Memory' + \
                   str(brain0.Q_eval.fc1_dims) + '-' + str(brain0.Q_eval.fc2_dims) + '.png'
        # todo: update when player turn implemented
        plotLearning(x, scores, eps_history, filename)
    if write_weights:
        modelname0 = "models/p{p}_q{q}_n{n}_k{k}_games{games}_fc1{fc1}_fc2{fc2}_alpha{alpha}_.pth" \
            .format(p=0, q=2, n=n, k=k, games=num_games, fc1=brain0.Q_eval.fc1_dims, fc2=brain0.Q_eval.fc2_dims,
                    alpha=brain0.Q_eval.alpha)
        brain0.save_model(modelname0)
