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


import copy
import warnings
import numpy as np
from scipy import stats as scipy_stats

from open_spiel.python import policy
from open_spiel.python.algorithms.sequence_form_utils import _EMPTY_INFOSET_ACTION_KEYS
from open_spiel.python.algorithms.sequence_form_utils import _EMPTY_INFOSET_KEYS
from open_spiel.python.algorithms.sequence_form_utils import _get_action_from_key
from open_spiel.python.algorithms.sequence_form_utils import construct_vars
from open_spiel.python.algorithms.sequence_form_utils import is_root
from open_spiel.python.algorithms.sequence_form_utils import policy_to_sequence
from open_spiel.python.algorithms.sequence_form_utils import sequence_to_policy
from open_spiel.python.algorithms.sequence_form_utils import uniform_random_seq
import pyspiel
import time

def neg_entropy(probs):
  return -scipy_stats.entropy(probs)


def softmax(x):
  unnormalized = np.exp(x - np.max(x))
  return unnormalized / np.sum(unnormalized)

def l2_norm(probs):
    return np.linalg.norm(probs) ** 2 / 2

def project(v, s = 1):
    # 将向量v欧几里得投影到概率单纯形上
    n_features = len(v)
    v_sorted = np.sort(v)[::-1]
    cssv = np.cumsum(v_sorted) - s
    ind = np.arange(n_features) + 1
    cond = v_sorted - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w / np.sum(w)

policy_threshold = 1e-50
def threshold(policy: np.array, mask: np.array) -> np.array:
  """Remove from the support the actions 'a' where policy(a) < threshold."""
  np.assert_equal_shape((policy, mask))
  if policy_threshold <= 0:
    return policy
  mask = mask * (
      (policy >= policy_threshold) +
      (np.max(policy, axis=-1, keepdims=True) < policy_threshold))
  return mask * policy / np.sum(mask * policy, axis=-1, keepdims=True)

def divergence(x, y, psi_x, psi_y, grad_psi_y):
  """Compute Bregman divergence between x and y, B_psi(x;y).

  Args:
      x: Numpy array.
      y: Numpy array.
      psi_x: Value of psi evaluated at x.
      psi_y: Value of psi evaluated at y.
      grad_psi_y: Gradient of psi evaluated at y.

  Returns:
      Scalar.
  """
  return psi_x - psi_y - np.dot(grad_psi_y, x - y)


def dilated_dgf_divergence(mmd_1, mmd_2):
  """Bregman divergence between two MMDDilatedEnt objects.

      The value is equivalent to a sum of two Bregman divergences
      over the sequence form, one for each player.

  Args:
      mmd_1: MMDDilatedEnt Object
      mmd_2: MMDDilatedEnt Object

  Returns:
      Scalar.
  """

  dgf_values = [mmd_1.dgf_eval(), mmd_2.dgf_eval()]
  dgf_grads = mmd_2.dgf_grads()
  div = 0
  for player in range(2):
    div += divergence(mmd_1.sequences[player], mmd_2.sequences[player],
                      dgf_values[0][player], dgf_values[1][player],
                      dgf_grads[player])
  return div


class MMDDilatedEnt(object):
  r"""Implements Magnetic Mirror Descent (MMD) with Dilated Entropy.

  The implementation uses the sequence form representation.

  The policies converge to a \alpha-reduced normal form QRE of a
  two-player zero-sum extensive-form game. If \alpha is set
  to zero then the method is equivalent to mirror descent ascent
  over the sequence form with dilated entropy and the policies
  will converge on average to a nash equilibrium with
  the appropriate stepsize schedule (or approximate equilirbrium
  for fixed stepsize).

  The main iteration loop is implemented in `update_sequences`:

  ```python
    game = pyspiel.load_game("game_name")
    mmd = MMDDilatedEnt(game, alpha=0.1)
    for i in range(num_iterations):
      mmd.update_sequences()
  ```
  The gap in the regularized game (i.e. 2x exploitability) converges
  to zero and can be computed:

  ```python
      gap = mmd.get_gap()
  ```
  The policy (i.e. behavioural form policy) can be retrieved:
  ```python
      policies = mmd.get_policies()
  ```

  The average sequences and policies can be retrieved:

  ```python
      avg_sequences = mmd.get_avg_sequences()
      avg_policies = mmd.get_avg_policies()
  ```

  """

  empy_state_action_keys = _EMPTY_INFOSET_ACTION_KEYS[:]
  empty_infoset_keys = _EMPTY_INFOSET_KEYS[:]

  def __init__(self, game, alpha, stepsize, itv, beta):
    """Initialize the solver object.

    Args:
        game: a zeros-um spiel game with two players.
        alpha: weight of dilated entropy regularization. If alpha > 0 MMD
          will converge to an alpha-QRE. If alpha = 0 mmd will converge to
          Nash on average.
        stepsize: MMD stepsize. Will be set automatically if None.
    """
    assert game.num_players() == 2
    assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
    assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
    assert (game.get_type().chance_mode
            == pyspiel.GameType.ChanceMode.DETERMINISTIC or
            game.get_type().chance_mode
            == pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC)
    assert alpha >= 0

    self.game = game
    self._cur_policy = policy.TabularPolicy(self.game)
    self.alpha = float(alpha)
    self.beta = float(beta)
    self.itv = itv
    (self.infosets, self.infoset_actions_to_seq, self.infoset_action_maps,
     self.infoset_parent_map, self.payoff_mat,
     self.infoset_actions_children) = construct_vars(game)
    self.last_gap = 0
    if stepsize is not None:
      self.stepsize = stepsize
    else:
      self.stepsize = self.alpha / (np.max(np.abs(self.payoff_mat))**2)

    if self.stepsize == 0.:
      warnings.warn("MMD stepsize is 0, probably because alpha = 0.")

    self.sequences = uniform_random_seq(game, self.infoset_actions_to_seq)
    self.ref_sequences= uniform_random_seq(game, self.infoset_actions_to_seq)
    self.avg_sequences = copy.deepcopy(self.sequences)
    self.iteration_count = 1
    self.grads_buffer = [[np.zeros(len(self.sequences[0]))], [np.zeros(len(self.sequences[1]))]]

  def current_policy(self):
    return self._cur_policy
  
  def get_parent_seq(self, player, infostate, sq):
    parent_isa_key = self.infoset_parent_map[player][infostate]
    seq_id = self.infoset_actions_to_seq[player][parent_isa_key]
    parent_seq = sq[player][seq_id]
    return parent_seq


  def get_infostate_seq(self, player, infostate):
    """Gets vector of sequence form values corresponding to a given infostate.

    Args:
        player: player number, either 0 or 1.
        infostate: infostate id string.

    Returns:
        Numpy array.
    """
    seq_idx = [
        self.infoset_actions_to_seq[player][isa_key]
        for isa_key in self.infoset_action_maps[player][infostate]
    ]
    seqs = np.array([self.sequences[player][idx] for idx in seq_idx])
    return seqs

  def dgf_eval(self):
    """Computes the value of dilated entropy for current sequences.

    Returns:
        List of values, one for each player.
    """
    dgf_value = [0., 0.]

    for player in range(2):
      for infostate in self.infosets[player]:

        if is_root(infostate):
          continue

        parent_seq = self.get_parent_seq(player, infostate)
        if parent_seq > 0:
          children_seq = self.get_infostate_seq(player, infostate)
          dgf_value[player] += parent_seq * neg_entropy(
              children_seq / parent_seq)

    return dgf_value

  def dgf_grads(self, pol, ref_pol):
    """Computes gradients of dilated entropy for each player and current seqs.

    Returns:
        A list of numpy arrays.
    """
    grads = [np.zeros(len(self.sequences[0])), np.zeros(len(self.sequences[1]))]
    ref_grads = [np.zeros(len(self.sequences[0])), np.zeros(len(self.sequences[1]))]
    for player in range(2):
      for infostate in self.infosets[player]:
        # infostates contain empty sequence for root variable
        if is_root(infostate):
          continue
        parent_seq = self.get_parent_seq(player, infostate, self.sequences)
        if parent_seq > 0:
          for isa_key in self.infoset_action_maps[player][infostate]:
            # compute infostate term
            seq_idx = self.infoset_actions_to_seq[player][isa_key]
            seq = self.sequences[player][seq_idx]
            grads[player][seq_idx] += seq / parent_seq
            # compute terms from children if there are any
            children = self.infoset_actions_children[player].get(isa_key, [])
            for child in children:
              child_isa_keys = self.infoset_action_maps[player][child]
              actions_child = [
                  _get_action_from_key(child_isa_key)
                  for child_isa_key in child_isa_keys
              ]
              policy_child = pol.policy_for_key(child)[:]
              policy_child = np.array([policy_child[a] for a in actions_child])              
              grads[player][seq_idx] -= l2_norm(policy_child)
        ref_parent_seq = self.get_parent_seq(player, infostate, self.ref_sequences)    
        if ref_parent_seq > 0:
          for isa_key in self.infoset_action_maps[player][infostate]:
            # compute infostate term
            seq_idx = self.infoset_actions_to_seq[player][isa_key]
            ref_seq = self.ref_sequences[player][seq_idx]
            ref_grads[player][seq_idx] += ref_seq / ref_parent_seq
            # compute terms from children if there are any
            children = self.infoset_actions_children[player].get(isa_key, [])
            for child in children:
              child_isa_keys = self.infoset_action_maps[player][child]
              actions_child = [
                  _get_action_from_key(child_isa_key)
                  for child_isa_key in child_isa_keys
              ]
              policy_child = ref_pol.policy_for_key(child)[:]
              policy_child = np.array([policy_child[a] for a in actions_child])              
              ref_grads[player][seq_idx] -= l2_norm(policy_child)
    return {"grads" : grads, "ref_grads" : ref_grads}

  def update_sequences(self):
    """Performs one step of MMD."""
    self.iteration_count += 1
    current_policy = self.get_policies()
    ref_policy = self.get_ref_policies()
    for player in range(2):
      psi_grads = self.dgf_grads(current_policy, ref_policy)
      bufferterm = np.sum(self.grads_buffer[player], axis=0)
      ori_grads = [self.payoff_mat @ self.sequences[1], self.payoff_mat.T @ self.sequences[0]]
      ori_grad = ori_grads[player]
      mmt_grads = (ori_grad - bufferterm * self.beta) * self.stepsize
      mmt_grads_player = [mmt_grads, -mmt_grads]
      # pylint: disable=invalid-unary-operand-type
      grads = (mmt_grads_player[player] - psi_grads["grads"][player] 
           - self.stepsize * self.alpha * psi_grads["ref_grads"][player]) / (1 + self.stepsize * self.alpha)
      
      self.grads_buffer[player].append(mmt_grads / self.stepsize)
      if self.itv > 0 and self.iteration_count % self.itv == 0:
        self.grads_buffer[player] = [np.zeros(len(self.sequences[player]))]
      new_policy = policy.TabularPolicy(self.game)
      self._update_state_sequences(self.empty_infoset_keys[player],
                                  grads, player, new_policy)
      self.sequences[player] = policy_to_sequence(self.game, new_policy,
                                          self.infoset_actions_to_seq)[player]
    if self.itv > 0 and self.iteration_count % self.itv == 0:
      self.ref_sequences = copy.deepcopy(self.sequences)     
    self.update_avg_sequences()

    
  def _update_state_sequences(self, infostate, g, player, pol):
    """Update the state sequences."""

    isa_keys = self.infoset_action_maps[player][infostate]
    seq_idx = [
        self.infoset_actions_to_seq[player][isa_key] for isa_key in isa_keys
    ]

    for isa_key, isa_idx in zip(isa_keys, seq_idx):

      # update children first if there are any
      children = self.infoset_actions_children[player].get(isa_key, [])
      for child in children:
        self._update_state_sequences(child, g, player, pol)
        # update gradient
        child_isa_keys = self.infoset_action_maps[player][child]
        child_seq_idx = [
            self.infoset_actions_to_seq[player][child_isa_key]
            for child_isa_key in child_isa_keys
        ]
        g_child = np.array([g[idx] for idx in child_seq_idx])

        actions_child = [
            _get_action_from_key(child_isa_key)
            for child_isa_key in child_isa_keys
        ]
        policy_child = pol.policy_for_key(child)[:]
        policy_child = np.array([policy_child[a] for a in actions_child])
        g[isa_idx] += np.dot(g_child, policy_child)
        g[isa_idx] += l2_norm(policy_child)

    # no update needed for empty sequence
    if is_root(infostate):
      return

    state_policy = pol.policy_for_key(infostate)
    g_infostate = np.array([g[idx] for idx in seq_idx])
    actions = [_get_action_from_key(isa_key) for isa_key in isa_keys]
    new_state_policy = project(-g_infostate)
    for action, pr in zip(actions, new_state_policy):
      state_policy[action] = max(pr, policy_threshold)

  def get_gap(self):
    """Computes saddle point gap of the regularized game.

    The gap measures convergence to the alpha-QRE.

    Returns:
        Scalar.
    """
    assert self.alpha > 0, "gap cannot be computed for alpha = 0"
    grads = [(self.payoff_mat @ self.sequences[1]) / (self.alpha),
             (-self.payoff_mat.T @ self.sequences[0]) / (self.alpha)]
    dgf_values = self.dgf_eval()

    br_policy = policy.TabularPolicy(self.game)
    for player in range(2):
      self._update_state_sequences(self.empty_infoset_keys[player],
                                   grads[player], player, br_policy)

    br_sequences = policy_to_sequence(self.game, br_policy,
                                      self.infoset_actions_to_seq)
    curr_sequences = copy.deepcopy(self.sequences)
    self.sequences = br_sequences
    br_dgf_values = self.dgf_eval()
    self.sequences = curr_sequences

    # gap of sequences (x,y)
    # d(x) + max_y' x.T A y'-d(y') + d(y) - min_x' d(x') + x'.T Ay

    gap = 0
    gap += curr_sequences[0].T @ self.payoff_mat @ br_sequences[1]
    gap += self.alpha * (dgf_values[1] - br_dgf_values[1])
    gap += self.alpha * (dgf_values[0] - br_dgf_values[0])
    gap += -br_sequences[0].T @ self.payoff_mat @ curr_sequences[1]
    return gap

  def update_avg_sequences(self):
    for player in range(2):
      self.avg_sequences[player] = self.avg_sequences[player] * (
          self.iteration_count - 1) + self.sequences[player]
      self.avg_sequences[
          player] = self.avg_sequences[player] / self.iteration_count

  def current_sequences(self):
    """Retrieves the current sequences.

    Returns:
      the current sequences for each player as list of numpy arrays.
    """
    return self.sequences

  def get_avg_sequences(self):
    """Retrieves the average sequences.

    Returns:
      the average sequences for each player as list of numpy arrays.
    """
    return self.avg_sequences

  def get_policies(self):
    """Convert current sequences to equivalent behavioural form policies.

    Returns:
        spiel TabularPolicy Object.
    """
    return sequence_to_policy(self.sequences, self.game,
                              self.infoset_actions_to_seq,
                              self.infoset_action_maps)
    
  def get_ref_policies(self):
    """Convert current sequences to equivalent behavioural form policies.

    Returns:
        spiel TabularPolicy Object.
    """
    return sequence_to_policy(self.ref_sequences, self.game,
                              self.infoset_actions_to_seq,
                              self.infoset_action_maps)

  def get_avg_policies(self):
    """Convert average sequences to equivalent behavioural form policies.

    Returns:
        spiel TabularPolicy Object.
    """
    return sequence_to_policy(self.avg_sequences, self.game,
                              self.infoset_actions_to_seq,
                              self.infoset_action_maps)
