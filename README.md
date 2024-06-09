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
- `Years`: Number of years the player has played in the major league.
- `League`: The league in which the player has played until the end of the season.

## Methodology
The project follows a structured methodology including:
1. **Data Preprocessing**: Handling missing values, outliers, and feature scaling.
2. **Feature Engineering**: Creating new features based on existing ones to enhance model performance.
3. **Model Selection**: Utilizing various machine learning algorithms such as Random Forest, Gradient Boosting, and CatBoost to train and evaluate the model.
4. **Hyperparameter Tuning**: Fine-tuning model hyperparameters to improve predictive accuracy.
5. **Model Evaluation**: Assessing model performance using metrics like RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).

## Usage
To replicate the analysis and train the model:
1. Clone this repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the Jupyter Notebook `baseball_salary_prediction.ipynb` to execute the code and train the model.

## Results
The trained machine learning model achieves an RMSE of [insert RMSE value] on the test data, demonstrating its effectiveness in predicting baseball player salaries based on their performance metrics.

## Future Work
Potential areas for future improvement and exploration include:
- Experimenting with additional machine learning algorithms.
- Incorporating more recent baseball player data to enhance model generalization.
- Exploring advanced feature engineering techniques to extract more meaningful insights from the data.

## Contributors
- John Doe (john.doe@example.com)
- Jane Smith (jane.smith@example.com)

## License
This project is licensed under the [MIT License](LICENSE).

