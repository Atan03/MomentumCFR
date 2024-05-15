# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example of sq_form_solver with dilated entropy to solve for QRE in Leduc Poker."""

from absl import app
from absl import flags
from open_spiel.python.algorithms import exploitability

import wandb
import sequence_form_algo.mmd_dilated as mmd_dilated
import sequence_form_algo.omwu_dilated as omwu_dilated
import sequence_form_algo.ogda_dilated as ogda_dilated
import sequence_form_algo.gda_dilated as gda_dilated
import sequence_form_algo.mmd_dilated_moving as mmd_dilated_moving
import sequence_form_algo.gda_dilated_moving as gda_dilated_moving
import sequence_form_algo.mommwu_dilated as mommwu_dilated
import sequence_form_algo.MoGDA_dilated as MoGDA_dilated
import pyspiel
import sys

flags.DEFINE_integer("iterations", 10000, "Number of iterations")
flags.DEFINE_float(
    "alpha", 0.0, "QRE parameter, larger value amounts to more regularization")
flags.DEFINE_string("game", "leduc_poker", "Name of the game")
flags.DEFINE_integer("print_freq", 10, "How often to print the gap")
flags.DEFINE_bool("use_wandb", True,
                  "use the policy finetune trick when evaluating.")
flags.DEFINE_string("project_name", "kuhn-poker vary k", "project name of wandb")
# flags.DEFINE_string("project_name", "liardice-5", "project name of wandb")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

if (FLAGS.use_wandb):
    wandb.init(
        project=FLAGS.project_name,
        config=FLAGS,
        name="MoGDA lr=2 beta=0.1 itv=50"
        # name="test"
    )
normalize = lambda x: (x - x.min()) / (x.max() - x.min())
# game_rps = np.array([
#     [0, 1, -1, 0, 0], 
#     [-1, 0, 1, 0, 0], 
#     [1, -1, 0, 0, 0], 
#     [1, -1, 0, -2, 1], 
#     [1, -1, 0, 1, -2], 
#     ])
def main(_):
  #
  
  # normalize = lambda x: (x - x.min()) / (x.max() - x.min())
  # seed = 0
  # size = 5
  # np.random.seed(seed)
  # W = np.random.randn(size, size)
  # S = np.random.randn(size, 1)
  # payoffs = 0.5 * (W - W.T) + S - S.T
  # payoffs /= np.abs(payoffs).max() 
  # payoffs_table = [normalize(payoffs).tolist(), (-normalize(payoffs)).tolist()]
  # game = pyspiel.create_matrix_game(payoffs_table[0],
  #                                   payoffs_table[1])
  # game = pyspiel.convert_to_turn_based(game)
  # game = pyspiel.load_game("battleship(board_height=2,board_width=2,ship_sizes=[2],num_shots=3,ship_values=[2],allow_repeated_shots=False)")
  # game = pyspiel.convert_to_turn_based(game)
  # game = pyspiel.load_game("turn_based_simultaneous_game(game=goofspiel(imp_info=True,num_cards=4,players=2,points_order=descending))")
  # game = pyspiel.load_game("liars_dice(dice_sides=5)")
  # game = pyspiel.load_game(FLAGS.game)
  game = pyspiel.load_game("kuhn_poker")
  
  
  # sq_form_solver = mmd_dilated.MMDDilatedEnt(game, alpha=0, stepsize=2)
  # sq_form_solver = gda_dilated.MMDDilatedEnt(game, alpha=0.0, stepsize=2)
  # sq_form_solver = omwu_dilated.MMDDilatedEnt(game, FLAGS.alpha, stepsize=2)
  # sq_form_solver = mmd_dilated_moving.MMDDilatedEnt(game, alpha=0.0005, stepsize=25, itv=500)
  # sq_form_solver = gda_dilated_moving.MMDDilatedEnt(game, alpha=0.0, stepsize=2, itv=20)
  # sq_form_solver = ogda_dilated.MMDDilatedEnt(game, alpha=0, stepsize=0.1, itv=1000000)
  # sq_form_solver = mommwu_dilated.MMDDilatedEnt(game, FLAGS.alpha, stepsize=2, beta=0.0, itv=50)
  sq_form_solver = MoGDA_dilated.MMDDilatedEnt(game, alpha=0, stepsize=2, beta=0.1, itv=50)
  for i in range(FLAGS.iterations):
    sq_form_solver.update_sequences()
    if i % FLAGS.print_freq == 0:
      conv = exploitability.nash_conv(game, sq_form_solver.get_policies())
      # conv = exploitability.nash_conv(game, mmd.get_avg_policies())
      # conv = mmd.get_gap()
      if (FLAGS.use_wandb):
            wandb.log({
                "conv": conv,
                "step": i + 1
            })
      print(conv)

if __name__ == "__main__":
  app.run(main)