import chess
import numpy as np
import pyspiel

from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv


class OpenSpielRbcEnv(OpenSpielEnv):

    def step(self, action):
        # Before applying action(s), there could be chance nodes.
        # E.g. if env has to figure out, which agent's action should get
        # resolved first in a simultaneous node.
        self._solve_chance_nodes()
        penalties = {}

        # Sequential game:
        if str(self.type.dynamics) == "Dynamics.SEQUENTIAL":

            curr_player = self.state.current_player()
            assert curr_player in action

            try:
                # TODO: (acxz) sometimes this is an illegal action yet no error is thrown
                # This is prob RBC code's fault for not throwing a pyspiel.SpielError i'm guessing
                assert action[curr_player] in self.state.legal_actions()

                # (acxz) take a look at self.state.apply_action_with_legality_check

                self.state.apply_action(action[curr_player])
            # TODO: (sven) resolve this hack by publishing legal actions
            #  with each step.
            except (AssertionError, pyspiel.SpielError) as e:
                self.state.apply_action(
                    np.random.choice(self.state.legal_actions()))
                penalties[curr_player] = -0.1

            # Compile rewards dict.
            rewards = {ag: r for ag, r in enumerate(self.state.returns())}
        # Simultaneous game.
        else:
            assert self.state.current_player() == -2
            # Apparently, this works, even if one or more actions are invalid.
            self.state.apply_actions(
                [action[ag] for ag in range(self.num_agents)])

        # Now that we have applied all actions, get the next obs.
        obs = self._get_obs()

        # Compile rewards dict and add the accumulated penalties
        # (for taking invalid actions).
        rewards = {ag: r for ag, r in enumerate(self.state.returns())}
        for ag, penalty in penalties.items():
            rewards[ag] += penalty

        # Are we done?
        is_done = self.state.is_terminal()

        dones = dict({ag: is_done
                      for ag in range(self.num_agents)},
                     **{"__all__": is_done})

        # return obs, rewards, dones, {}
        # (acxz)
        # I should extend the env wrapper
        # locally with my mods till i get things working
        # Let's add the state in infos!
        # openspiel has self.state.information_state_tensor(), but not all games implement it
        if self.state.is_terminal():
            # terminal state empty => infos empty
            infos = {}
        else:
            # SEQUENTIAL_GAMES
            infos = {self.state.current_player(): {"state": self.state}}
        # TODO need to handle case for SIMULATANEOUS GAMES
        # RBC is not simultaneous so don't worry for now

        return obs, rewards, dones, infos

    def render(self, mode=None) -> None:
        if mode == "human":
            print(self.state)
            # TODO: print previous action need to do all that prev state
            # action_to_string conversion
            # but makes the games much more readable
            print(chess.Board(self.state.__str__()))
            print()
