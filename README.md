# rl-match-prediction-j48-c4.5

This Python program is designed to predict the outcome of games (win or loss) based on a dataset containing various game and player statistics. It employs a Decision Tree Classifier for the prediction, which is similar to the J48 model in Weka (an implementation of the C4.5 algorithm).

## How to Run the Program

1. Ensure that you have Python installed on your system. If not, you can download it from [python.org](https://www.python.org/downloads/).
2. Install the required libraries (if you haven't already) by running:
   ```
   pip install pandas scikit-learn
   ```
3. Download the `game_outcome_prediction.py` file and the dataset.
4. Run the program using:
   ```
   python game_outcome_prediction.py
   ```

## Program Workflow

1. The program starts by loading a dataset of game and player statistics.
2. It then preprocesses this data by encoding categorical variables.
3. The data is split into a training set and a test set.
4. A Decision Tree Classifier is trained using the training data.
5. The classifier then makes predictions on the test data.
6. Finally, the program prints the accuracy and a detailed classification report.

## Dataset

The dataset should be in CSV format and contain various game and player statistics, along with the game outcomes (win/loss). The program is specifically tailored for a dataset with a structure similar to the one used in the example (`stats.csv`).

## Output

The program outputs the accuracy of the model and a detailed classification report that includes precision, recall, and F1-score for each class (win and loss).

## Customization

You can customize this program to suit different datasets or to experiment with other machine learning models by modifying the `game_outcome_prediction.py` file.
