Problem Statement:
Every March, millions of people fill out brackets trying to predict the outcome of the NCAA basketball tournament. This competition formalises that challenge — given decades of historical game data, predict the win probability for every possible matchup in the 2026 Men's and Women's tournaments.
Predictions are evaluated using the Brier Score — the mean squared error between your predicted probability and the actual outcome (1 = win, 0 = loss). Lower is better. A model that always predicts 50/50 scores 0.25; a competitive model typically scores around 0.17–0.21.
The challenge is that tournament upsets are common and genuinely hard to predict. The data is rich but noisy, and the tournament itself only has ~67 games per year to validate against.


Dataset
All data is provided by Kaggle and covers both Men's and Women's NCAA basketball. Key files used:
https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data


Men's TeamIDs range from 1000–1999. Women's TeamIDs range from 3000–3999. This is used to split the submission file into two halves and predict each with its own model.


Approach
Matchup Framing
Every historical tournament game is converted into a labelled row. By convention, the team with the lower TeamID is always T1 and the higher is T2. Label = 1 if T1 won, 0 if T2 won. This matches the submission file format exactly (SSSS_XXXX_YYYY where XXXX < YYYY).
For each matchup we compute the difference between T1's features and T2's features. A positive difference means T1 is stronger on that metric. This framing gives the model a clean signal without needing to understand absolute team quality.
Features Engineered
From regular season box scores:

Average points scored, field goal %, 3-point %, free throw %, total rebounds, assists, turnovers, steals, blocks, fouls

From regular season results:

Win rate
Average point differential (margin of victory weighted across all games)
Last 10 games win rate (current form heading into the tournament)
Neutral court experience rate (fraction of games played on neutral courts)

From tournament seeds:

Numeric seed (1–16) for each team
Seed difference between the two teams

From Massey Ordinals (Men's only):

Average rank across all ~40 ranking systems at the pre-tournament snapshot (DayNum=133)
Best rank given by any single system

From conference data:

Conference tournament wins and win rate (current form signal)
Conference strength — historical tournament win rate of the team's conference

From coaches file (Men's only):

Number of prior NCAA tournament appearances by the head coach

From game cities + Cities.csv:

Travel diversity — number of different states a team played in during the regular season

From secondary tournaments:

Number of prior NIT / secondary tournament appearances

From historical tournament box scores:

Cumulative average tournament performance in prior seasons (shooting %, scoring, rebounding etc.)

Missing Values
Several features are legitimately missing for many teams — for example, a team that lost in the first round of their conference tournament has no conference tournament win rate, and early seasons have no prior tournament history to look back on. Rather than dropping rows (which wipes all training data), missing values are filled with sensible defaults:

Win rates and percentages → 0.5 (no information, assume average)
Ranking columns → median rank
Count and experience columns → 0

Model
LightGBM with 5-fold cross-validation. One model trained for Men's, one for Women's (since Women's lacks Massey Ordinals and coach data).
Predictions from all 5 fold models are averaged (ensemble) and clipped to [0.025, 0.975] to avoid extreme Brier score penalties for confident wrong predictions.
LightGBM was chosen because:

Handles missing values and mixed-quality features well
Fast to train on tabular data
Less prone to overfitting than deep models given the small tournament dataset (~1,000–2,000 labelled games per gender)


