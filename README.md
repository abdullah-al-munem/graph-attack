# Graph Attack

Graph Attack is a Python project that implements a proposed attack model on graph data. 

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Flowchart](#flowchart)


## Introduction

A targeted attack on Node classification on Graph Dataset.

## Features

The attack have two module.

- Add edge module
- Remove edge module

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
2. To perform the **proposed attack**, run the following command:
   ```bash
    python proposed_attack_model.py
   ```
3. To perform a single **state of the art** attack, use **state_of_the_art_attack_models.py** script and edit accordingly.
    ```bash
    python state_of_the_art_attack_models.py
   ```
4. To perform **all the attacks**, run the following command:
   ```bash
    python perform_all_attacks.py
   ```
   or,
   ```bash
    python perform_all_attacks.py --dataset citeseer --defense_model mdgcn
   ```

<comment> ## Flowchart

<comment> Check out the flowchart PDF in the [flowchart folder](./flowchart/).

<comment> [![Flowchart](./flowchart/Graph_Attack_Module.png)](./flowchart/Graph_Attack_Module.pdf)




