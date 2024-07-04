import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from enum import Enum, StrEnum

# Define string enumerations for the columns
class EventColumns(StrEnum):
    EVENT_TYPE = 'event_type'
    PLAYER = 'player'
    PLAYER2 = 'player2'
    COUNTRY = 'country'
    IS_GOAL = 'is_goal'
    LOCATION = 'location'
    BODYPART = 'bodypart'
    ASSIST_METHOD = 'assist_method'
    SITUATION = 'situation'

# Define the Events enumeration
class Events(Enum):
    SHOT = 1

# Class for the xG model
class XGModel:
    """Class to handle the xG (expected Goals) model and data processing."""

    # Get the directory path of the current script
    CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    DATA_DIRECTORY = os.path.join(CURRENT_DIRECTORY, '..', 'data')
    MODEL_DIRECTORY = os.path.join(CURRENT_DIRECTORY, '..', 'models')

    def __init__(self):
        """Initialize the XGModel class."""
        self.model = None
        self.shots = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self._read_data()

    
    def test_model(self):
        """Test the xG model and print the confusion matrix."""

        # Ensure that the model is trained before testing
        if self.model is None:
            print("Error: Model not trained. Please train the model before testing.")
            return

        # Predict using the trained model
        y_pred = self.model.predict(self.X_test)

        # Get the confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Print the confusion matrix
        print("Confusion Matrix:")
        print(cm)

    def _read_data(self):
        """Reads the CSV files, merges data, and preprocesses the DataFrame."""

        # Read the CSV files
        events = pd.read_csv(os.path.join(self.DATA_DIRECTORY, 'events.csv'))
        info = pd.read_csv(os.path.join(self.DATA_DIRECTORY, 'ginf.csv'))

        # Merge the required info from the games into the events dataframe
        events = events.merge(info[['id_odsp', EventColumns.COUNTRY.value, 'date']], on='id_odsp', how='left')

        # Extract the year from the date
        events['year'] = [datetime.strptime(x, "%Y-%m-%d").year for x in events['date']]

        # Create the shots DataFrame and update the strings to title case
        self.shots = events[events[EventColumns.EVENT_TYPE.value] == Events.SHOT.value]
        self.shots.loc[:, EventColumns.PLAYER.value] = self.shots[EventColumns.PLAYER.value].str.title()
        self.shots.loc[:, EventColumns.PLAYER2.value] = self.shots[EventColumns.PLAYER2.value].str.title()
        self.shots.loc[:, EventColumns.COUNTRY.value] = self.shots[EventColumns.COUNTRY.value].str.title()

        # Prepare data for training the xG model

        # Create binary representations of these columns
        data = pd.get_dummies(self.shots.iloc[:, -8:-3], columns=[
                              EventColumns.LOCATION.value, EventColumns.BODYPART.value,
                              EventColumns.ASSIST_METHOD.value, EventColumns.SITUATION.value])
        
        # These columns are already binary
        data.columns = ['fast_break', 'loc_centre_box', 'loc_diff_angle_lr', 'diff_angle_left', 'diff_angle_right',
                        'left_side_box', 'left_side_6ybox', 'right_side_box', 'right_side_6ybox', 'close_range',
                        'penalty', 'outside_box', 'long_range', 'more_35y', 'more_40y', 'not_recorded', 'right_foot',
                        'left_foot', 'header', 'no_assist', 'assist_pass', 'assist_cross', 'assist_header',
                        'assist_through_ball', 'open_play', 'set_piece', 'corner', 'free_kick']

        # Finally add the value that determines wether a shot was successful or not
        data[EventColumns.IS_GOAL.value] = self.shots[EventColumns.IS_GOAL.value]

        # Split the dataset into training and test sets
        self.X = data.iloc[:, :-1]
        self.y = data.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.35, random_state=1)

        # Load or train the xG model
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Loads a pre-existing model if it exists, otherwise trains a new one."""

        # Predefine the later used model
        self.model = None

        # Get a list of files in the model folder
        model_list = os.listdir(self.MODEL_DIRECTORY)

        # Take the first .pkl file from the list
        pkl_file = next((file for file in model_list if file.endswith('.pkl')), None)

        # Check if a .pkl file was found
        if pkl_file is not None:
            # Load the first .pkl file
            with open(os.path.join(self.MODEL_DIRECTORY, pkl_file), 'rb') as f:
                self.model = pickle.load(f)
                print("xG-Model has been loaded successfully.")
        else:
            # Train the xG model if no .pkl file found
            self._train_model()

    def _train_model(self):
        """Trains a new GradientBoostingClassifier model using hyperparameter optimization."""

        def evaluate_model(params):
            """Evaluate the model using hyperparameter values."""
            model = GradientBoostingClassifier(
                learning_rate=params['learning_rate'],
                min_samples_leaf=params['min_samples_leaf'],
                max_depth=params['max_depth'],
                max_features=params['max_features']
            )

            model.fit(self.X_train, self.y_train)
            return {
                'loss': 1 - roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1]),
                'status': STATUS_OK,
                'model': model
            }

        # Define the hyperparameter space for hyperopt
        hyperparameter_space = {
            'learning_rate': hp.uniform('learning_rate', 0.05, 0.3),
            'min_samples_leaf': hp.choice('min_samples_leaf', range(15, 200)),
            'max_depth': hp.choice('max_depth', range(2, 20)),
            'max_features': hp.choice('max_features', range(3, 27))
        }

        # Initialize hyperopt trials and perform hyperparameter optimization
        trials = Trials()
        best = fmin(
            evaluate_model,
            space=hyperparameter_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials
        )

        # Get the best model from the trials
        self.model = trials.best_trial['result']['model']

        # Save the trained model
        with open(os.path.join(self.MODEL_DIRECTORY, 'xG_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        print("xG-Model has been trained and saved successfully.")


    def get_shots(self):
        """
        This function returns the shots DataFrame
        """
        # Create a copy to avoid SettingWithCopyWarning
        shots = self.shots.copy()

        # Predict expected goals using the model
        shots['expectedGoals'] = self.model.predict_proba(self.X)[:, 1]
        shots['expectedGoals'] = round(shots['expectedGoals'], 2)

        return shots



    def get_xg_dataframe(self):
        """Returns a DataFrame with xG statistics for top players."""

        # Create a copy to avoid SettingWithCopyWarning
        shots = self.shots.copy()

        # Predict expected goals using the model
        shots['expectedGoals'] = self.model.predict_proba(self.X)[:, 1]

        # Calculate the difference between expected goals and actual goals
        shots['difference'] = shots['expectedGoals'] - shots[EventColumns.IS_GOAL.value]

        # Group and calculate player statistics
        players = shots.groupby(EventColumns.PLAYER.value).sum().reset_index()
        players.rename(columns={EventColumns.IS_GOAL.value: 'trueGoals'}, inplace=True)
        players['expectedGoals'] = round(players['expectedGoals'], 2)
        players['difference'] = round(players['difference'], 2)
        players['ratio'] = players['trueGoals'] / players['expectedGoals']

        return players



if __name__ == "__main__":
    # Create the xG model object
    xg_model = XGModel()

    # Get the DataFrame with xG statistics
    xg_model.test_model()