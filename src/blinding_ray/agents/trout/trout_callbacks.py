# Callbacks that the Trout Policy requires

import os
import psutil
from typing import Dict

import chess
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
from ray.train import Trainer
from reconchess.bots.trout_bot import STOCKFISH_ENV_VAR


class TroutCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy], episode: Episode,
                         **kwargs) -> None:

        # get trout policy
        # TODO move trout_id to self.trout_id
        trout_id = 'trout'
        trout_policy = policies[trout_id]

        # Create internal representation of the board to keep track of
        trout_policy.board = chess.Board()

        # use policy mapping to get the color
        white_player = episode.policy_mapping_fn(
            int(chess.WHITE), episode, worker)
        black_player = episode.policy_mapping_fn(
            int(chess.BLACK), episode, worker)

        if trout_id == white_player:
            trout_policy.color = chess.WHITE
        if trout_id == black_player:
            trout_policy.color = chess.BLACK

        # TODO: stockfish creation should just happen in policy init
        # no need to do this every time episode starts
        # problem: how to kill the engine via this method
        # need to kill it when the policy class ends

        # One benefit of this method is that if stockfish dies in the previous
        # episode, we can restart stockfish on episode start

        # make sure stockfish environment variable exists
        if STOCKFISH_ENV_VAR not in os.environ:
            raise KeyError(
                'TroutBot requires an environment variable called "{}" pointing to the Stockfish executable'.format(
                    STOCKFISH_ENV_VAR))

        # make sure there is actually a file
        stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        if not os.path.exists(stockfish_path):
            raise ValueError(
                'No stockfish executable found at "{}"'.format(stockfish_path))

        # initialize the stockfish engine
        trout_policy.engine = chess.engine.SimpleEngine.popen_uci(
            stockfish_path, setpgrp=True)

        # Reset variables required for certain functionality
        trout_policy.prev_state = None
        trout_policy.first_move = True

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy], episode: Episode,
                       **kwargs) -> None:

        # TODO: do we even need to kill stockfish at the end of episode just
        # end of training right?

        # get trout policy
        trout_id = 'trout'
        trout_policy = policies[trout_id]

        try:
            # if the engine is already terminated then this call will throw an exception
            trout_policy.engine.quit()
            print("killing the fish")
        except chess.engine.EngineTerminatedError:
            pass

    def on_train_result(self, *, trainer: "Trainer", result: dict,
                        **kwargs) -> None:

        # find and kill the last stockfish process

        # get all children processes of current program run
        children_processes = psutil.Process().children()

        # find stockfish engine pid
        engine_pid = None
        for children_process in children_processes:
            if children_process.name() == "stockfish":
                engine_pid = children_process.pid

        try:
            engine_process = psutil.Process(engine_pid)
            engine_process.terminate()
            print("banishing the fish")
        except:
            print("prob already dead")
