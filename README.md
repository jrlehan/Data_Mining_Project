# Poker Decision Analysis

## Project Overview

This project analyzes a large-scale poker decision dataset to understand patterns in optimal (GTO-based) post-flop strategy. The primary goal is to examine how game-state variables influence the hero’s optimal action and to evaluate whether these relationships can be modeled using data mining techniques.

In addition to exploratory data analysis (EDA), this project includes formal research question formation and structured methodological planning using both course-based and external statistical techniques.

---

## Research Questions

### RQ1: Predictive Modeling
**Which game-state variables most effectively predict the GTO-approved decision (check, call, bet, raise, fold)?**

- **Method:** Decision Tree Classification (Course Technique)
- **Goal:** Identify interpretable splits that approximate optimal strategy.
- **Evaluation:** Accuracy, tree depth, information gain, interpretability.

---

### RQ2: Pattern Discovery
**What recurring strategic patterns appear in the dataset, and do they reveal structural tendencies or potential selection bias?**

- **Method:** Frequent Itemset Mining (Apriori – Course Technique)
- **Goal:** Identify high-support combinations of position, street, and action.
- **Evaluation:** Support, confidence, lift, interpretability.

---

### RQ3: Statistical Association
**Is there a statistically significant association between hero position and the GTO-approved decision?**

- **Method:** Chi-Squared Test of Independence (External Technique)
- **Goal:** Test whether position and optimal decision are independent.
- **Evaluation:** Chi-square statistic, p-value, Cramér’s V (effect size).

---

## Dataset

The dataset consists of decision points for the hero in poker hands. Each row represents a single post-flop evaluation and includes structured game-state information and the labeled optimal action.

### Core Columns

| Column | Description |
|--------|-------------|
| preflop_action | Betting actions before the flop |
| board_flop | Flop cards (3 cards) |
| board_turn | Turn card |
| board_river | River card |
| aggressor_position | Position of the aggressor |
| postflop_action | Betting sequence after the flop |
| evaluation_at | Street of decision (Flop/Turn/River) |
| available_moves | Legal actions for hero |
| pot_size | Current pot size |
| hero_position | Hero’s position (IP/OOP) |
| holding | Hero’s hole cards |
| correct_decision | GTO-labeled optimal decision |
| Index | Original row index |

---

## Data Types

- Categorical sequences
- Text-encoded action tokens
- Numerical variables (e.g., pot size)
- Categorical labels (correct decision)

---

## Feature Engineering

Additional features created during analysis include:

- Hand strength indicators (pair, suitedness, rank gap)
- Board texture features (paired board, connectivity, etc.)
- Pot size buckets (Small, Medium, Large, Huge)
- Simplified decision categories (Passive vs Aggressive)

These features support predictive modeling, pattern mining, and statistical testing.

---

## Methodological Approach

This project combines:

### Course Techniques
- Decision Trees
- Apriori Frequent Itemset Mining

### External Techniques
- Chi-Squared Test of Independence
- Effect Size Analysis (Cramér’s V)

### Evaluation Strategy
- Model performance metrics (accuracy, interpretability)
- Pattern quality metrics (support, confidence, lift)
- Statistical significance testing
- Baseline comparisons (e.g., majority-class predictor)

---

## Exploratory Data Analysis (EDA)

The EDA examines:

- Distribution of decisions across streets
- Position-based decision patterns
- Pot size relationships
- Board texture influence
- Class balance of optimal decisions
- Association between position and action

Initial statistical testing indicates a strong association between hero position and optimal decision, supporting the feasibility of RQ3.

---

## Notebook Contents

The accompanying notebook includes:

1. **Data Loading & Cleaning**
   - CSV import
   - Column renaming
   - Type conversions
   - Missing value handling

2. **Feature Engineering**
   - Hand-based features
   - Board texture extraction
   - Decision simplification
   - Pot size categorization

3. **Exploratory Data Analysis**
   - Distribution plots
   - Heatmaps
   - Cross-tabulations
   - Statistical testing

4. **Method Feasibility Testing**
   - Decision tree training
   - Apriori execution
   - Chi-square testing
   - Package validation

---

## Usage

To use this project:

1. Clone the repository.
2. Upload the dataset (.csv) to the same directory. A link to the dataset can be found in the notebook.
3. Open the notebook in Google Colab or Jupyter.
4. Run all cells sequentially to reproduce the EDA and RQ explorations.

---

## Next Steps

Future extensions may include:

- Random forest comparison
- Sequential pattern mining
- Deep learning-based modeling
- Cross-validation and hyperparameter tuning
- Model complexity vs interpretability analysis
- Deeper text embedding exploration

---

## Note

This project is intended for educational and analytical purposes and does not provide real-world poker strategy recommendations.
