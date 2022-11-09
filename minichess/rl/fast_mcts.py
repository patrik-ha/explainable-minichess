from shutil import move
from minichess.agents.lite_model import LiteModel
from minichess.agents.predictor_convnet import PredictorConvNet
from minichess.chess.move_utils import calculate_all_moves, index_to_move
from minichess.rl.chess_helpers import get_initial_chess_object, get_settings, launch_tensorboard, random_string
from montecarlo.node import Node
from montecarlo.montecarlo import MonteCarlo
from minichess.rl.case_utils import checkpoint, prepare_distribution
from minichess.agents.convnet import ConvNet
from multiprocessing import Pool
import numpy as np
import os
import gc
from minichess.chess.fastchess_utils import inv_color, visualize_board


def perform_mcts_episodes(args):

    np.seterr(over="ignore", invalid="raise")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    episodes, full_name, model_name, sim_steps, prior_noise_coefficient, sample_ratio, cpuct, nondet_plies, thread = args
    state_buffer = []
    distribution_buffer = []
    value_buffer = []
    game_winners = []

    episode_game = get_initial_chess_object(full_name)
    dims = episode_game.dims

    agent = PredictorConvNet(LiteModel.from_file("minichess/agents/checkpoints/{}/{}/temp.tflite".format(full_name, model_name)))

    all_moves, all_moves_inv = calculate_all_moves(dims)
    move_cap = all_moves_inv.shape[0]

    for episode in range(episodes):
        print("Thread {} starting, episode {}.".format(thread, episode + 1))
        episode_game = get_initial_chess_object(full_name)
        states = []
        distributions = []
        turns = []
        root = Node(episode_game, None, move_cap, all_moves, all_moves_inv, nondet_plies, cpuct)
        montecarlo = MonteCarlo(root, all_moves, move_cap, dims, prior_noise_coefficient, cpuct, agent)
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
            # visualize_board(episode_game.bitboards, episode_game.dims)

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
    # print("Thread {} finished.".format(thread))
    return state_buffer, distribution_buffer, value_buffer, game_winners


import sys

if __name__ == "__main__":
    np.seterr(over="ignore")
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    settings_path = os.path.join(os.getcwd(), sys.argv[1])
    if not os.path.exists(settings_path):
        raise Exception("Settings file does not exist")
    settings = get_settings(settings_path)

    USE_TENSORBOARD = settings["use_tensorboard"]

    EPOCHS = settings["epoch_cap"]
    SIM_STEPS = settings["simulation_steps"]
    REPLAY_BUFFER_CAP = settings["replay_buffer_size"]
    SAMPLE_RATIO = settings["sample_ratio"]

    full_name = settings["board_name"]
    model_name = settings["model_name"]

    THREADS = settings["threads"]
    EPISODES_PER_THREAD_INSTANCE = settings["episodes_per_thread"]

    CHECKPOINT_INTERVAL = settings["checkpoint_interval"]
    EPOCH_CHECKPOINTS_TO_SKIP = settings["epoch_checkpoints_to_skip"]

    PRIOR_NOISE_COEFFICIENT = settings["prior_noise_coefficient"]
    CPUCT = settings["cpuct"]
    NONDET_PLIES = settings["nondet_plies"]

    episode_game = get_initial_chess_object(full_name)
    dims = episode_game.dims

    all_moves, all_moves_inv = calculate_all_moves(dims)
    move_cap = all_moves_inv.shape[0]

    agent = ConvNet(episode_game.agent_board_state().shape, move_cap)

    state_buffer = []
    distribution_buffer = []
    value_buffer = []
    os.makedirs("minichess/agents/checkpoints/{}/{}".format(full_name, model_name), exist_ok=True)
    if USE_TENSORBOARD:
        base_summary_dir = os.path.join(os.getcwd(), "tensorboard_logs", full_name, model_name)
        full_summary_dir = os.path.join(base_summary_dir, random_string(5))
        summary_writer = tf.summary.create_file_writer(full_summary_dir)
        launch_tensorboard(base_summary_dir)
    checkpoint_path = checkpoint(0, agent.model, full_name, model_name, None)
    for epoch in range(1, EPOCHS + 1):
        with open("minichess/agents/checkpoints/{}/{}/temp.tflite".format(full_name, model_name), "wb") as f:
            lite_model = LiteModel.from_keras_model_as_bytes(agent.model)
            f.write(lite_model)
        # results = perform_mcts_episodes((
        #     EPISODES_PER_THREAD_INSTANCE,
        #     full_name,
        #     model_name,
        #     SIM_STEPS,
        #     PRIOR_NOISE_COEFFICIENT,
        #     SAMPLE_RATIO,
        #     CPUCT,
        #     NONDET_PLIES,
        #     1
        # ))
        # results = [results]
        with Pool(THREADS) as p:
            results = p.map(perform_mcts_episodes, [
                (
                    EPISODES_PER_THREAD_INSTANCE,
                    full_name,
                    model_name,
                    SIM_STEPS,
                    PRIOR_NOISE_COEFFICIENT,
                    SAMPLE_RATIO,
                    CPUCT,
                    NONDET_PLIES,
                    thread
                ) for thread in range(THREADS)])
        singular_outcomes = []
        for thread_result in results:
            state_buffer.extend(thread_result[0])
            for dist in thread_result[1]:
                distribution_buffer.append(dist / np.sum(dist))
            value_buffer.extend(thread_result[2])
            singular_outcomes.extend(thread_result[3])
        singular_outcomes = np.array(singular_outcomes)

        if USE_TENSORBOARD:
            with summary_writer.as_default():
                tf.summary.text("Simulation game results", "W-D-B: {}-{}-{}".format((singular_outcomes == 1).sum(), (singular_outcomes == 2).sum(), (singular_outcomes == 0).sum()), step=epoch)
        # Generate stuff for three epochs before starting to checkpoint and train
        history = None
        if epoch > EPOCH_CHECKPOINTS_TO_SKIP:
            state_buffer = state_buffer[-REPLAY_BUFFER_CAP:]
            distribution_buffer = distribution_buffer[-REPLAY_BUFFER_CAP:]
            value_buffer = value_buffer[-REPLAY_BUFFER_CAP:]
            history = agent.fit(np.array(state_buffer), np.array(distribution_buffer), np.array(value_buffer), epochs=1)

            if USE_TENSORBOARD:
                with summary_writer.as_default():
                    for loss in ["loss", "value_output_loss", "policy_output_loss"]:
                        tf.summary.scalar(name=loss, data=history.history[loss][0], step=epoch)
                summary_writer.flush()

        if (epoch % CHECKPOINT_INTERVAL) == 0 or epoch == 1:
            checkpoint_path = checkpoint(epoch, agent.model, full_name, model_name, history)
            if USE_TENSORBOARD:
                with summary_writer.as_default():
                    tf.summary.text("Checkpoints", "Checkpoint made at {}.".format(checkpoint_path), step=epoch)
                summary_writer.flush()
