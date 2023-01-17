import numpy as np
from minichess.rl.chess_helpers import get_initial_chess_object

def save_root_info(agent, full_name, model_name, epoch):
    base_game = get_initial_chess_object(full_name)
    root_priors, root_value = agent.predict(base_game.agent_board_state())
    suffix = "0" if (epoch % 2) == 0 else "1"
    inv_suffix = "0" if (epoch % 2) != 0 else "1"
    np.savez("minichess/agents/checkpoints/{}/{}/latest{}.npz".format(full_name, model_name, suffix), priors=root_priors[0], root_value=root_value[0])
    if epoch == 0:
        np.savez("minichess/agents/checkpoints/{}/{}/latest{}.npz".format(full_name, model_name, inv_suffix), priors=root_priors[0], root_value=root_value[0])

def load_root_info(full_name, model_name):
    # With a large amount of threads, a thread might try to read the root-info while the main thread
    # is writing to it.
    # Stupid way to fix this, but write to two of them, and make sure that the main thread
    # only writes to one of them each epoch.
    try:
        data = np.load("minichess/agents/checkpoints/{}/{}/latest0.npz".format(full_name, model_name))
    except:
        data = np.load("minichess/agents/checkpoints/{}/{}/latest1.npz".format(full_name, model_name))
    return data["root_value"], data["priors"]