# Baseball Players Salary Prediction

![Baseball](imagine.jpg)

## Overview
This project aims to develop a machine learning model to predict the salaries of baseball players based on their past performance. The dataset used for this task includes statistics from the 1986-1987 season, sourced from Carnegie Mellon University's StatLib library. Various machine learning algorithms are employed to train the model and make accurate salary predictions.

## Dataset
The dataset consists of 20 variables and 322 observations, with a total size of 21 KB. Some key features include:
- `AtBat`: Number of hits with a baseball bat during the 1986-1987 season.
- `Hits`: Number of hits during the 1986-1987 season.
- `HmRun`: Number of home runs during the 1986-1987 season.
- `Runs`: Number of runs scored for the team during the 1986-1987 season.
- `RBI`: Number of runs batted in during the 1986-1987 season.
- `Walks`: Number of walks drawn by the player during the 1986-1987 season.
- `Years`: Number of years the player has played in the major league.
- `CAtBat`: Number of times the player has been at bat during their career.
- `CHits`: Number of hits the player has made during their career.
- `CHmRun`: Number of home runs the player has made during their career.
- `CRuns`: Number of runs the player has scored during their career.
- `CRBI`: Number of runs batted in by the player during their career.
- `CWalks`: Number of walks drawn by the player during their career.
- `League`: The league in which the player has played until the end of the season, represented by factors A and N.
- `Division`: The division in which the player played at the end of the 1986 season, represented by factors E and W.
- `PutOuts`: Number of times a player assisted another player in making a putout during the 1986-1987 season.
- `Assists`: Number of assists made by the player during the 1986-1987 season.
- `Errors`: Number of errors made by the player during the 1986-1987 season.
- `Salary`: The player's salary for the 1986-1987 season (in thousands of dollars).
- `NewLeague`: The league in which the player started the 1987 season, represented by factors A and N.

## Methodology
The project follows a structured methodology including:
1. **Exploratory Data Analysis**:This stage involves examining the dataset in depth, including an overall view, analysis of categorical and numerical variables, assessment of the target variable, and correlation analysis.
2. **Data Preprocessing**: Handling missing values, outliers, and feature scaling.
3. **Feature Engineering**: Creating new features based on existing ones to enhance model performance.
4. **Model Selection**: Utilizing various machine learning algorithms such as Random Forest, Gradient Boosting, and CatBoost to train and evaluate the model.
5. **Hyperparameter Tuning**: Fine-tuning model hyperparameters to improve predictive accuracy.
6. **Model Evaluation**: Assessing model performance using metrics like RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).



## Results
### Random Forest Regressor (RF)
- RF best params: {'max_depth': 15, 'max_features': 7, 'min_samples_split': 8, 'n_estimators': 500}
- RMSE (After): 214.9493 (RF)

### Gradient Boosting Regressor (GBM)
- GBM best params: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.7}
- RMSE (After): 205.1289 (GBM)

### LightGBM Regressor
- LightGBM best params: {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'n_estimators': 500}
- RMSE (After): 223.4899 (LightGBM)

### CatBoost Regressor
- CatBoost best params: {'depth': 6, 'iterations': 500, 'learning_rate': 0.1}
- RMSE (After): 219.5299 (CatBoost)

### General Conclusion
- **Gradient Boosting Regressor (GBM)** demonstrated the strongest predictive performance with the lowest RMSE value of 205.1289.
- **Random Forest Regressor (RF)** also showed competitive performance, achieving an RMSE of 214.9493.
- **LightGBM** and **CatBoost Regressors** provided valuable insights into salary prediction despite slightly higher RMSE values.
- Considering the salary range variability in the dataset (min: $67,500, max: $2,460,000), the obtained RMSE values are generally small and acceptable.
- These findings highlight the potential of machine learning models in accurately predicting baseball player salaries, aiding informed decision-making in player contract negotiations and team management.
- Future research could explore advanced feature engineering techniques and alternative modeling approaches to enhance predictive accuracy.



## Future Work
Potential areas for future improvement and exploration include:
- Experimenting with additional machine learning algorithms.
- Incorporating more recent baseball player data to enhance model generalization.
- Exploring advanced feature engineering techniques to extract more meaningful insights from the data.



## License
This project is licensed under the [MIT License](LICENSE).

