import numpy as np
import os
from minichess.chess.move_utils import move_to_index
import json


def prepare_distribution(dist, board_dimensions, move_cap, all_moves, turn):
    """Returnerer en fordeling over besøk, slik at dist[flat_trekk] = visits for hvert lovlige trekk i posisjonen. 0 for ikke-lovlige trekk."""
    full_move_distributions = np.zeros((board_dimensions[0], board_dimensions[1], move_cap))
    for (move, visit_count) in dist:
        (i, j), (dx, dy), promotion = move
        move_index = move_to_index(all_moves, dx, dy, promotion, turn)
        full_move_distributions[i, j, move_index] = visit_count
    # full_move_distributions /= np.sum(full_move_distributions)
    return full_move_distributions.flatten()


def checkpoint(epoch, net, full_name, agent_name, history=None):
    os.makedirs("minichess/agents/checkpoints/{}/{}".format(full_name, agent_name), exist_ok=True)
    checkpoint_path = "minichess/agents/checkpoints/{}/{}/{}".format(full_name, agent_name, epoch)
    net.save(checkpoint_path)
    net.save_weights(checkpoint_path + ".h5")
    if history is not None:
        with open("minichess/agents/checkpoints/{}/{}/{}.json".format(full_name, agent_name, epoch), "w") as f:
            f.write(json.dumps(history.history))
    return checkpoint_path
