# Explainable Minichess

## What is this?
This is a codebase that provides:
- A fast and adaptable chess-simulator for Python, designed to be used with Reinforcement Learning loops and MCTS
- A fully-functioning training loop for training neural-net chess agents on smaller chess variants, following AlphaZero.
- Functionality for detecting pre-defined concepts in the trained neural chess agents.

## Installation and setup
1. Clone the repo.
2. Run (alternatively inside a virtual environment) `python -m pip install -r requirements.txt`

## Usage
The codebase comes with some pre-defined chess variants. These are the standard 8x8-version of chess (called `8x8standard`), and 5x4- and 6x6-variants. (called `5x4silverman` and `6x6losalomos`, respectively).
### Defining your own board
You can also create your own variant, if you have some specific variaint that you want to explore and train models on.
#### The board
These are defined in [/minichess/boards](/minichess/boards). Here, the file `*.board` implicitly defines the starting-position and the board size. If you want to create your own variant, you simply create another file in this folder (`/minichess/boards/YOUR_CHESS_NAME_HERE.board`). An example of the standard 8x8 chess position is shown below:

```
rnbqkbnr
pppppppp
        
        


PPPPPPPP
RNBQKBNR
```
Here, the characters `[r, n, b, q, k]` designate a "rook", "knight", "bishop", "queen",and "king", respectively. If these are upper-case, they are expected to be pieces belonging to the player playing as White, and if lower-case, they belong to the player playing as Black.

#### Castling
The file `*.castle` defines castling for your specific variant. If your variant doesn't support castling, you don't have to create the file.
The file supports up to two castling-variants, to the "left", and to the "right". In practice, this means moving the king two squares to the left, or right, and moving the rook to the outside. This means that the configurability for castling is somewhat limited, but from what I could see, most smaller variants that were applicable here either had this type of castling, or didn't support castling at all. For the standard 8x8 chess position, it looks like this.
```
01110000
00111000

00000110
00001110
```
Here, each pair of two lines, separated by an empty line, defines a castling variant. The first line defines what squares must be unoccupied for castling to be valid, and the second line defines what squares must be unattacked for castling to be valid.


### Configuration alternatives for training
If you want to train your own agents, many parts of the training loop can be configured. An example is located in [/configs/base.json](/configs/base.json).

The syntax of this file is:
```javascript
{
    // The board-variant to use for training
    "board_name": "5x4silverman", 
     // What to call the agent (used for folders, etc)
    "model_name": "target_net_test"
    // Whether to use the resnet-architecture (defined in minichess/agents/resnet.py), or the standard-architecture (defined in minichess/agents/convnet.py)
    "use_resnet": true, 
    // How many threads to use for simulation
    "threads": 2, 
    // How many episodes to play out per thread
    "episodes_per_thread": 2, 
    // Size of the replay buffer used to train the agents after each epoch
    "replay_buffer_size": 30000,
    // How many simulation-iterations to take for each "move" in each episode 
    "simulation_steps": 200, 
    // How many epochs (episodes * threads) to train for
    "epoch_cap": 1500, 
    // How often to save a checkpoint of the agent, '1' means after every epoch, '2' means after every second epoch, etc.
    "checkpoint_interval": 1, 
    // How many epochs to skip training for at the beginning of the training phase (this is useful if you want to let the replay buffer fill a bit before starting to train your agent)
    "epoch_checkpoints_to_skip": 0, 
    // How large the ratio is for the noise that is added to the prior noise when exploring the search tree
    "prior_noise_coefficient": 0.2, 
    // Search coeffient, see AlphaZero-paper
    "cpuct": 1.4, 
    // The ratio of positions to sample, per game, on average
    "sample_ratio": 0.2, 
    // Having the first n plies of the episode game being selected by a weighted choice, instead of argmax 
    "nondet_plies": 8, 
    // Using tensorboard (path is printed to output when starting, defaults to localhost:6006)
    "use_tensorboard": true
}
```
### Training
Starting the training loop with this configuration file is done by running:

`python -m minichess.rl.fast_mcts configs/base.json`
## Detecting concepts
After you have trained some agents, (or using the pre-trained ones), you can detect concepts from these by running [/detect_concepts_bulk.ipynb](/detect_concepts_bulk.ipynb). This produces the raw concept-detection data, in the folder [/concept_presences](/concept_presences).
The graphs are then produced by running [/concept_bootstrap_surface.ipynb](/concept_bootstrap_surface.ipynb) with the name of the model specified at the top of the notebook.
For larger models this is quite resource-intensive, and often requires a lot of VRAM. 
## Using the simulator on its own
This repo provides a relatively fast simulation environment for Python for smaller chess variants. The board size, castling rules, and starting position are fully customizable. The logic for converting moves to/from a neural network policy format is built on top, so the simulator itself can also be detached.
