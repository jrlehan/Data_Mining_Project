# Poker Decision Analysis

## Project Overview
This project explores a dataset of poker hand decisions, focusing on the hero’s actions at various points during a hand. The goal is to understand patterns in decision-making based on:

- Hero position (in-position vs out-of-position)  
- Hand strength (pocket pairs, suitedness, etc.)  
- Board texture (paired, connected, two-tone flops)  
- Pot size  
- Street (flop, turn, river)  

The project is currently at the **exploratory data analysis (EDA) stage**, with visualizations and feature engineering to identify key trends in poker strategy.

---

## Dataset

The dataset consists of decision points for the hero in poker hands. Each row represents a single decision and includes the following columns:

| Column | Description |
|--------|-------------|
| preflop_action | Betting actions before the flop |
| board_flop | Flop cards (3 cards) |
| board_turn | Turn card |
| board_river | River card |
| aggressor_position | Position of the current aggressor |
| postflop_action | Betting actions after the flop |
| evaluation_at | Street at which the hero’s decision occurs (Flop/Turn/River) |
| available_moves | List of legal actions for the hero |
| pot_size | Current pot size at the decision point |
| hero_position | Hero’s position (IP/OOP) |
| holding | Hero’s hole cards |
| correct_decision | Labeled optimal decision |
| Index | Original row index (optional) |

Additional features are created in the EDA notebook for analysis, including:

- Hero hand features (`is_pair`, `is_suited`, `rank_gap`, etc.)  
- Board texture features (`flop_paired`, `flop_connected`, `flop_two_tone`)  
- Pot size bucket (`Small`, `Medium`, `Large`, `Huge`)  
- Bet size relative to pot (`decision_bucket`)  

---

## Notebook

The accompanying notebook contains:

1. **Data Loading & Cleaning**  
   - Reads CSV into pandas DataFrame  
   - Handles missing values and column renaming  

2. **Feature Engineering**  
   - Hand features, board texture, pot buckets, bet size relative to pot  
   - Positional indicators (IP/OOP)  

3. **Exploratory Data Analysis (EDA)**  
   - Distribution of decisions across streets, positions, hand types, board textures, and pot sizes  
   - Heatmaps and bar plots to visualize strategy patterns  
   - Analysis of bet size buckets relative to pot and strategic implications  

---

## Usage

To use this project:

1. Clone the repository and upload the dataset (`.csv`) to the same directory.  
2. Open the notebook in Google Colab or Jupyter.  
3. Run all cells sequentially to reproduce the exploratory data analysis and visualizations.  

---

## Next Steps

Future work may include:

- Encoding `postflop_action` into structured features for modeling  
- Predictive modeling of hero decisions using decision trees, classification models, or deep learning  
- Analysis of optimal strategy patterns based on position, hand strength, and board texture  

---

**Note:** This project is for educational and analytical purposes and does not provide real-world poker strategy recommendations.  
