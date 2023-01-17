from shutil import move
from minichess.agents.lite_model import LiteModel
from minichess.agents.predictor_convnet import PredictorConvNet
from minichess.chess.move_utils import calculate_all_moves, index_to_move
from minichess.rl.chess_helpers import get_initial_chess_object, get_settings
from montecarlo.node import Node
from montecarlo.montecarlo import MonteCarlo
from minichess.rl.case_utils import checkpoint, prepare_distribution
from minichess.agents.convnet import ConvNet
from minichess.agents.resnet import ResNet
from multiprocessing import Pool
import numpy as np
import os
import gc
from minichess.chess.fastchess_utils import inv_color, visualize_board
from mpi4py import MPI
from montecarlo.predictor import Predictor
from minichess.rl.caches import save_root_info, load_root_info

from minichess.rl.shard_storage import ShardStorage


def perform_mcts_episodes(args):
    np.seterr(over="ignore", invalid="raise")
    episodes, full_name, model_name, sim_steps, prior_noise_coefficient, sample_ratio, cpuct, nondet_plies, thread = args
    state_buffer = []
    distribution_buffer = []
    value_buffer = []
    game_winners = []

    episode_game = get_initial_chess_object(full_name)
    dims = episode_game.dims

    root_value, root_priors = load_root_info(full_name, model_name)

    all_moves, all_moves_inv = calculate_all_moves(dims)
    move_cap = all_moves_inv.shape[0]
    root = Node(episode_game, None, move_cap, all_moves, all_moves_inv, nondet_plies, cpuct)
    montecarlo = MonteCarlo(thread, root, all_moves, move_cap, dims, prior_noise_coefficient, cpuct, root_priors, root_value)
    print("Thread {} starting shard.".format(thread), flush=True)
    for episode in range(episodes):
        episode_game = get_initial_chess_object(full_name)
        root = Node(episode_game, None, move_cap, all_moves, all_moves_inv, nondet_plies, cpuct)
        montecarlo.move_root(root)
        states = []
        distributions = []
        turns = []
        
        current_ply = 0
        while episode_game.game_result() is None:

            for s in range(sim_steps):
                montecarlo.simulate()

            selected_child = montecarlo.make_choice(current_ply)
            move_to_make = selected_child.move_indices
            distribution = montecarlo.distribution()
            i, j, ind = move_to_make
            dx, dy, promotion = index_to_move(all_moves_inv, ind, episode_game.turn)

            # Record distribution and board BEFORE making the move.
            if np.random.random() < sample_ratio:
                states.append(episode_game.agent_board_state())
                distributions.append(distribution.flatten())
                turns.append(episode_game.turn)

            episode_game.make_move(i, j, dx, dy, promotion)
            montecarlo.move_root(selected_child)

            gc.collect()

            # if the outcome is -1, that means that the person who has the current turn has lost
            # So it needs to be flipped back before starting to save
            outcome = episode_game.game_result()
            current_ply += 1
        # Don't look too closely at this...
        # Essentially, 2 is a draw, 1 is a win for the player whose turn it is, -1 is a loss for the player whose turn it is
        if outcome == 0:
            game_winner = 2
        else:
            game_winner = 1 if outcome == 1 else 0

        for (dist, state, turn) in zip(distributions, states, turns):
            if game_winner == 2:
                outcome == 0
            if game_winner == turn:
                outcome = 1
            if game_winner != turn:
                outcome = -1
            state_buffer.append(state)
            distribution_buffer.append(dist)
            value_buffer.append(outcome)
        game_winners.append(game_winner)
        del episode_game
        del root
        gc.collect()
        montecarlo.episode_done()
    montecarlo.all_done()
    return state_buffer, distribution_buffer, value_buffer, game_winners


import sys

if __name__ == "__main__":
    def get_agent(use_resnet, init=True, compile_model=True):
        if use_resnet:
            agent = ResNet(episode_game.agent_board_state().shape, move_cap, init=init, compile_model=compile_model)
        else:
            agent = ConvNet(episode_game.agent_board_state().shape, move_cap, init=init, compile_model=compile_model)
        return agent

    comm = MPI.COMM_WORLD
    GPU_THREAD = 1
    MAIN_THREAD = 0
    rank = comm.Get_rank()
    ranksize = comm.Get_size()
    amount_of_gpus = 1
    np.seterr(over="ignore")
    if rank == 0 or rank == 1:
        from minichess.rl.tensorboard_writer import TensorboardWriter
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    settings_path = os.path.join(os.getcwd(), sys.argv[1])
    if not os.path.exists(settings_path):
        raise Exception("Settings file does not exist")
    settings = get_settings(settings_path)

    USE_TENSORBOARD = settings["use_tensorboard"]

    USE_RESNET = settings["use_resnet"]
    EPOCHS = settings["epoch_cap"]
    SIM_STEPS = settings["simulation_steps"]
    REPLAY_BUFFER_CAP = settings["replay_buffer_size"]
    SAMPLE_RATIO = settings["sample_ratio"]

    full_name = settings["board_name"]
    model_name = settings["model_name"]

    EPISODES_PER_REFRESH = settings["episodes_per_refresh"]
    EPISODES_PER_THREAD_INSTANCE = EPISODES_PER_REFRESH // (ranksize - 2)

    CHECKPOINT_INTERVAL = settings["checkpoint_interval"]
    EPOCH_CHECKPOINTS_TO_SKIP = settings["epoch_checkpoints_to_skip"]

    PRIOR_NOISE_COEFFICIENT = settings["prior_noise_coefficient"]
    CPUCT = settings["cpuct"]
    NONDET_PLIES = settings["nondet_plies"]

    episode_game = get_initial_chess_object(full_name)
    dims = episode_game.dims

    all_moves, all_moves_inv = calculate_all_moves(dims)
    move_cap = all_moves_inv.shape[0]

    if rank == 0:
        os.makedirs("minichess/agents/checkpoints/{}/{}".format(full_name, model_name), exist_ok=True)
        shard_storage = ShardStorage(REPLAY_BUFFER_CAP)
        tensorboard_writer = TensorboardWriter(USE_TENSORBOARD, full_name, model_name)
        agent = get_agent(USE_RESNET)
        checkpoint_path = checkpoint(0, agent.model, full_name, model_name, None)
        save_root_info(agent, full_name, model_name, 0)
    
    comm.Barrier()
    if rank >= 2:
        while True:
            results = perform_mcts_episodes((
                EPISODES_PER_THREAD_INSTANCE,
                full_name,
                model_name,
                SIM_STEPS,
                PRIOR_NOISE_COEFFICIENT,
                SAMPLE_RATIO,
                CPUCT,
                NONDET_PLIES,
                rank
            ))

            data = {
                "states": results[0],
                "distributions": results[1],
                "values": results[2],
                "winners": results[3]
            }
            print("Thread {} finishing shard.".format(rank))
            comm.send(data, 0)
            del data
            del results
    if rank == 1:
        print("Predictor starting.")
        agent = get_agent(USE_RESNET, True, False)
        agent.load_weights(0, full_name, model_name)
        predictor = Predictor(ranksize - 2, episode_game.agent_board_state().shape, agent, full_name, model_name)
        while True:
            predictor.work()


    if rank == 0:
        epoch = 0
        while True:
            while True:
                results = comm.recv()
                # print("A result is received!", flush=True)
                shard_storage.extend_storage(results["states"], results["distributions"], results["values"], results["winners"])
                if shard_storage.shards_received != 0 and ((shard_storage.shards_received % EPISODES_PER_REFRESH) == 0):
                    break
            epoch += 1
        

            tensorboard_writer.write_outcomes(shard_storage.outcomes, epoch)                
            # Might want to generate training data for some epochs before actually starting to train
            if epoch > EPOCH_CHECKPOINTS_TO_SKIP:
                shard_storage.truncate_storage()
                history = agent.fit(*shard_storage.get_training_samples(), epochs=1)
                tensorboard_writer.write_losses(history, epoch)
            
            checkpoint(epoch, agent.model, full_name, model_name, None)
            tensorboard_writer.write_checkpoint_message(checkpoint_path, epoch)

            save_root_info(agent, full_name, model_name, epoch)

            print("Telling predictor to load weights.")
            buf = np.array([epoch])
            res = comm.Isend([buf, MPI.FLOAT], 1, tag=7)
            res.wait()


