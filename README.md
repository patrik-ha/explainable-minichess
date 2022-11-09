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
The codebase comes with some pre-defined chess variants. These are the standard 8x8-version of chess (called "`8x8standard`"), and 5x4- and 6x6-variants. (called `"5x4silverman"` and `6x6losalomos`, respectively).
### Defining your own board
You can also create your own variant, if you have some specific variaint that you want to explore and train models on.
#### The board
These are defined in [/minichess/boards](/minichess/boards). Here, the file "`*.board`"" implicitly defines the starting-position and the board size. If you want to create your own variant, you simply create another file in this folder ("`/minichess/boards/YOUR_CHESS_NAME_HERE.board`"). An example of the standard 8x8 chess position is shown below:

```
rnbqkbnr
pppppppp
        
        


PPPPPPPP
RNBQKBNR
```
Here, the characters `[r, n, b, q, k]` designate a "rook", "knight", "bishop", "queen",and "king", respectively. If these are upper-case, they are expected to be pieces belonging to the player playing as White, and if lower-case, they belong to the player playing as Black.

#### Castling
The file "`*.castle`"" 

### Configuration alternatives for training
If you want to train your own agents, many parts of the training loop can be configured. An example is located in [/configs/base.json](/configs/base.json).

The syntax of this file is:
```
{
    "board_name": "5x4silverman", // The board-variant to start each 
    "model_name": "target_net_test",
    "threads": 2,
    "episodes_per_thread": 2,
    "replay_buffer_size": 30000,
    "simulation_steps": 200,
    "epoch_cap": 1500,
    "checkpoint_interval": 1,
    "epoch_checkpoints_to_skip": 0,
    "prior_noise_coefficient": 0.2,
    "cpuct": 1.4,
    "sample_ratio": 0.2,
    "nondet_plies": 8,
    "use_tensorboard": true
}

```


## Detecting concepts
TODO

## Using the simulator on its own
This repo provides a relatively fast simulation environment for Python for smaller chess variants. The board size, castling rules, and starting position are fully customizable.

## Reproducing results
This codebase is part of an article (xx.xx.xx.xx), and the code to reproduce the results shown is detailed below. 
### Detecting concepts
The agents that were used for concept-detection is included in the repo.
One can detect concepts from these by running [/detect_concepts_bulk.ipynb](/detect_concepts_bulk.ipynb). This produces the raw concept-detection data, in the folder [/concept_presences](/concept_presences).
The graphs are then produced by running [/concept_bootstrap_surface.ipynb](/concept_bootstrap_surface.ipynb) with the name of the model specified at the top of the notebook.

