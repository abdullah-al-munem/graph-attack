# GAEttack

GAEttack: Graph Auto-Encoder based Adversarial Attacks on Graph Neural Networks

## Abstract

## Features

- Add edge
- Remove edge

## Installation

1. **Python Version**: This project requires Python **3.8.0**. If you don't have it installed, you can download it from [Python's official website](https://www.python.org/downloads/release/python-380/).
2. Create a virtual environment using the following command:
   ```bash
    py -3.10 -m venv attack_venv
   ```
   or,
   ```bash
    conda create -n attack_venv python=3.10 -y
   ```
3. **Dependencies**: Install dependencies after activating the virtual environment:
   ```bash
   pip install -r requirements.txt
   ```
4. **Additional Dependencies**: Install additional dependencies:
   ```bash
   python install_reqs.py
   ```

## Usage

1. Activate the virtual environment.
2. Precompute Important Edge Set: Edit the code accordingly (specify the dataset name).
   ```bash
    python create_important_edge_set.py
   ```
3. Precompute Distance JSON: Edit the code accordingly (specify the dataset name).
   ```bash
    python precompute_distance.py
   ```
4. To perform the **proposed attack**, run the following command:
   ```bash
    python perform_single_attack.py --surrogate_model gcn --dataset cora --defense_model gcn --budget 1 --target_node 1687
   ```
   or, edit the proposed_attack_model.py file accordingly and run it.
   ```bash
    python proposed_attack_model.py
   ```
5. To perform a single **state of the art** attack, use **state_of_the_art_attack_models.py** script and edit accordingly.
   ```bash
   python state_of_the_art_attack_models.py
   ```
6. To perform **all the attacks**, run the following command:
   ```bash
    python perform_all_attacks.py
   ```
   or,
   ```bash
    python perform_all_attacks.py --dataset citeseer --defense_model mdgcn
   ```

<!-- ## Flowchart -->

<!-- Check out the flowchart PDF in the [flowchart folder](./flowchart/). -->

<!-- [![Flowchart](./flowchart/Graph_Attack_Module.png)](./flowchart/Graph_Attack_Module.pdf) -->
