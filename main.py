from board import Board
from game import Game
from random_play import Randy
from phils_dq import TF_Player, Agent


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
        scores = []
        eps_history = []
        score = 0
    for i in range(num_games):
        if meta:
            pass  # print stuff to stdout every 10 episodes, every episode, whatever
            # do some other stuff like updating eps_history
        board.reset()
        done = False
        score = 0
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
