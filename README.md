# Roots of Risk: Modeling Poker Strategy

## Project Overview
This project analyzes a large-scale poker decision dataset to understand patterns in optimal (GTO-based) post-flop strategy. The primary goal is to examine how game-state variables influence the hero’s optimal action and to evaluate whether these relationships can be modeled using data mining techniques. Furthermore, this project aims to explore the tradeoff between explainable models and high-performing models. If successful, the results of this project could be transformed into a curriculum to develop a poker player, increasing their ability to win money at a growing game.

The main deliverable for this project is main_notebook.ipynb. A project pitch can be accessed at: https://youtu.be/q106wLPUXNA
---

## Research Questions

1. Does incorporating more data improve a decision tree's accuracy in predicting GTO-approved decisions?
2. Which game-state variables most effectively predict the GTO-approved decision?

---

## Data

I used the PokerBench dataset, which contains 500k post-flop decision points for the hero in poker hands. Each row represents a single post-flop evaluation and includes structured game-state information and the labeled optimal action.

The dataset can be accessed here: https://huggingface.co/datasets/RZ412/PokerBench. Specifically, the postflop_500k_train_set_game_scenario_information.csv was used.

During preprocessing, I simplified the correct_decision column. Rather than show a specific amount to bet or raise, all bets / raises were condensed into a single "Bet/Raise" label.

I implemented additional preprocessing and feature engineering for each player profile built. The exact preprocessing per profile is described in main_notebook.ipynb.

---

## Results Summary

Poker can be studied through a visualizable decision tree in order to beat basic LLMs. Additional practice could lead to the skills required to outperform trained LLMs.

---

## Steps to Reproduce

This project was built in Google Colab with Python 3.12.13. All package versions are pinned in requirements.txt, exported from the Colab session this work was developed in.

The pipeline has three phases: install dependencies, run preprocessing once, then run the analysis notebook. Preprocessing is separated from the notebook so the notebook itself stays focused on the analysis story — every step just loads a preprocessed dataframe and trains a model.

Set up a virtual environment and install dependencies:

bash   python3.12 -m venv .venv
   source .venv/bin/activate              # macOS / Linux
   .venv\Scripts\activate                 # Windows PowerShell
   pip install -r requirements-local.txt
   pip install jupyter                    # not in requirements-local.txt; Colab provides this for free

Place the raw PokerBench CSV at data/raw/postflop_500k_train_set_game_scenario_information.csv.
Download at: https://huggingface.co/datasets/RZ412/PokerBench.

Run preprocessing once:
bash   python -m scripts.build_features data/raw/postflop_500k_train_set_game_scenario_information.csv.

Launch Jupyter from the project root and open the notebook:
bash   jupyter notebook
Click main_notebook.ipynb in the file listing, then Cell → Run All.

Optional: validation and unit tests
After preprocessing has produced the parquet files, you can verify the pipeline:

Module-level unit tests (data-independent, run in seconds):

bash  python -m src.hand_evaluator     # 19 tests + theoretical 7,462-class check
  python -m src.max_strength       # 12 tests on hand-crafted scenarios
  python -m src.nut_strength       # 5 tests on hand-crafted scenarios

Data-dependent invariants on the full 500k rows: open and run notebooks/validation.ipynb.

Re-running
scripts/build_features.py overwrites the parquet files each run, so it's safe to re-run any time you've changed something under src/. The main notebook never does any preprocessing, so iterating on the analysis is fast — no recompute needed unless features change.

---

## Key Dependencies

- Python 3.12.13 (used to run code)
- scikit-learn 1.6.1 (used for basic decision tree models, accuracy reporting)
- xgboost 3.2.0 (used for advanced decision tree models)
- matplotlib 3.10.0 (used to plot decision trees and feature importance)

---

## Repository Structure

```
./
├── checkpoints/
│   ├── Project_Checkpoint_1.ipynb
│   ├── Project_Checkpoint_2.ipynb
├── data/                                       # only exists at the local level
│   ├── raw/                                    # upload raw csv here
│   └── processed/                              # scripts will place processed data here
├── notebooks/                                  # contains iterations of the main_notebook
│   ├── Project_Main_Middle_Draft.ipynb
│   ├── Project_Main_Rough_Draft.ipynb
│   ├── validation.ipynb                        # contains validations ran on main_notebook data
├── scripts/
│   ├── __init__.py
│   ├── build_features.py                       # ran in order to produce processed data
├── src/                                        # contains helper functions for main_notebook preprocessing
│   ├── __init__.py
│   ├── action_parser.py
│   ├── cards.py
│   ├── features.py
│   ├── hand_evaluator.py
│   ├── max_strength.py
│   ├── nut_strength.py
├── main_notebook.ipynb
├── requirements.txt
├── requirements-local.txt
└── README.md
```

---