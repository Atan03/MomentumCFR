# Copyright 2019 DeepMind Technologies Limited
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

"""Tests for open_spiel.python.algorithms.discounted_cfr."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.algorithms import discounted_cfr
from open_spiel.python.algorithms import cfr
import MoCFR 
import pyspiel
from open_spiel.python.algorithms import exploitability
import wandb
from absl import flags
import sys
FLAGS = flags.FLAGS
flags.DEFINE_bool("use_wandb", True,
                  "use the policy finetune trick when evaluating.")
flags.DEFINE_string("project_name", "leduc-poker", "project name of wandb")
FLAGS(sys.argv)

if (FLAGS.use_wandb):
    wandb.init(
        project=FLAGS.project_name,
        config=FLAGS,
        name="MoCFR+ beta=0.01 itv=30"
    )
    
EFGDATA = """
  EFG 2 R "RPS" { "Player 1" "Player 2" } ""
  p "ROOT" 1 1 "P1 Infoset" { "R" "P" "S" } 0
    p "R" 2 1 "P2 Infoset" { "R" "P" "S" } 0
      t "RR" 1 "Outcome RR" { 0.0 0.0 }
      t "RP" 2 "Outcome RP" { -1.0 1.0 }
      t "RS" 3 "Outcome RS" { 3.0 -3.0 }
    p "P" 2 1 "P2 Infoset" { "R" "P" "S" } 0
      t "PR" 4 "Outcome PR" { 1.0 -1.0 }
      t "PP" 5 "Outcome PP" { 0.0 0.0 }
      t "PS" 6 "Outcome PS" { -1.0 1.0 }
    p "S" 2 1 "P2 Infoset" { "R" "P" "S" } 0
      t "SR" 7 "Outcome SR" { -3.0 3.0 }
      t "SP" 8 "Outcome SP" { 1.0 -1.0 }
      t "SS" 9 "Outcome SS" { 0.0  0.0 }
"""
game = pyspiel.load_game("leduc_poker")
# game = pyspiel.load_game_as_turn_based("matrix_rps")
# game = pyspiel.load_efg_game(EFGDATA)
# game = pyspiel.load_game("turn_based_simultaneous_game(game=goofspiel(imp_info=True,num_cards=5,players=2,points_order=descending))")
# game = pyspiel.load_game("liars_dice(dice_sides=5)")
# game = pyspiel.load_game("battleship(board_height=2,board_width=2,ship_sizes=[2],num_shots=3,ship_values=[2],allow_repeated_shots=False)")
# solver = discounted_cfr.DCFRSolver(game)
solver = MoCFR.CFRPlusSolver(game, itv=30, mu=0.01)
# solver = MoCFR_local.CFRPlusSolver(game, itv=100, mu=0.003, alter=True)
# solver = RTCFR.CFRPlusSolver(game, itv=50, mu=0.5)
# solver = regCFR_ori.CFRPlusSolver(game, itv=25, mu=0.6)
# solver = cfr.CFRSolver(game)
# solver = cfr.CFRPlusSolver(game)
# solver = pcfr.CFRPlusSolver(game)

for i in range(50000):
    solver.evaluate_and_update_policy()
    if i % 10 == 0:
        # conv = exploitability.nash_conv(game, solver.average_policy())
        conv = exploitability.nash_conv(game, solver.current_policy())
        print(conv)
        if (FLAGS.use_wandb):
            wandb.log({
                "conv": conv,
                "step": i + 1
            })

