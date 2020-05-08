carlos.py contains tests to find how best to create a Monte Carlo-style playout. The "meat" of the implementation is in board.py.

demo.py was used for the oral exam.

eve.py (as opposed to PvP or PvE) is used to train the RL agent against itself and/or random/greedy play.

phils_utils.py and dq.py are adapted from the Github page referenced in the paper.

game.py and env.py are relics of a simpler time, and their use is deprecated (but, as all superstitious developpers do, we keep them there... *just in case*).

main.py contains the exact setup used to prove the property that in 3^3 4-player Tic-Tac-Toe it is possible for players 2,3, and 4 to ally against 1 and force 1 to lose. Note that due to the probabilistic nature of the agent, it is possible the experiment will need to be rerun a few times before the same result is obtained.
> also contains examples to perform similar searches with RL as a guide and a probabilistic guide

cimpl contains files from when C implementation was being considered and tested.

obsolete contains deprecated files, if of interest.

runtime contains scripts to compare the efficiency with which different approaches performed the gather() operation.

models contains saved RL neural network models.
