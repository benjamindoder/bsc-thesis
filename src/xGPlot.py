import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xGModel import *

class ExpectedGoalsPlot:
    """
    This class plots different xG statistics for players.
    """

    def __init__(self):
        """
        Initializes the object and fetches the xG data from XGModel.
        """
        self.xg_model = XGModel()
        self.xg_df = self.xg_model.get_xg_dataframe()

    def plot_best_xg_goal_diff(self):
        """
        Plots the top players based on the difference between actual goals scored and expected goals.
        """

        # Sort by xG to goal difference
        xg_goal_df = self.xg_df.sort_values(['difference', 'trueGoals']).reset_index(drop=True)

        # Add a Ranking column
        xg_goal_df['rank'] = xg_goal_df.index + 1

        # Only save the top 10 players
        xg_goal_df = xg_goal_df[['rank', EventColumns.PLAYER.value, 'difference', 'trueGoals', 'expectedGoals', 'ratio']].head(10)

        self._plot_bar(xg_goal_df, 'difference', 'player', "Top Overachievers: Players with most goals compared to xG",
                       'Difference between xG and goals scored', x_range=np.arange(0, 65, 5))

    def plot_worst_xg_goal_diff(self):
        """
        Plots the bottom players based on the difference between actual goals scored and expected goals.
        """

        # Sort by xG to goal difference
        xg_goal_df = self.xg_df.sort_values(['difference', 'trueGoals']).reset_index(drop=True)

        # Add a ranking column
        xg_goal_df['rank'] = xg_goal_df.index + 1

        # Only save the bottom 10 players
        xg_goal_df = xg_goal_df[['rank', EventColumns.PLAYER.value, 'difference', 'trueGoals', 'expectedGoals', 'ratio']].tail(10)

        # Reverse the order of rows
        xg_goal_df = xg_goal_df.iloc[::-1]

        self._plot_bar(xg_goal_df, 'difference', 'player', "Bottom Underachievers: Players with fewest goals compared to xG",
                       'Difference between xG and goals scored')


    def plot_best_ratio(self):
        """
        Plots the top players based on the ratio of actual goals scored to expected goals.
        """

        # Sort by xG to goal ratio descending
        xg_ratio_df = self.xg_df[self.xg_df['trueGoals']>30].sort_values(['ratio', 'trueGoals'], ascending=False).reset_index(drop=True)

        # Add a ranking column
        xg_ratio_df['rank'] = xg_ratio_df.index + 1

        # Only save the top 10 players
        xg_ratio_df = xg_ratio_df[['rank', 'player', 'ratio', 'trueGoals', 'expectedGoals']].head(10)

        self._plot_bar(xg_ratio_df, 'ratio', 'player', "Best Finishers: goals/xG", 'Goals scored per one xG')


    def plot_worst_ratio(self):
        """
        Plots the bottom players based on the ratio of actual goals scored to expected goals.
        """

        # Sort by xG to goal ratio ascending
        xg_ratio_df = self.xg_df[self.xg_df['trueGoals']>30].sort_values(['ratio', 'trueGoals'], ascending=True).reset_index(drop=True)

        # Add a ranking column
        xg_ratio_df['rank'] = xg_ratio_df.index + 1

        # Get the first 10 rows of the DataFrame
        xg_ratio_df = xg_ratio_df.head(10)

        self._plot_bar(xg_ratio_df, 'ratio', 'player', "Worst Finishers: goals/xGoals", 'Goals Scored per one XGoal')

    
    def plot_most_xg(self):
        """
        Plots the top player with the highest amount of expected goals
        """
        
        # Sort by expected goals and only take the top 10 players
        total_xg = self.xg_df.sort_values(by='expectedGoals', ascending=False).head(10)

        self._plot_bar(total_xg, 'expectedGoals', 'player', "Highest amount of expected goals", 'Expected Goals')

    
    def plot_top_xg_shots_ratio(self):
        """
        Plots the top players based on the ratio of expected goals to the number of shots taken.
        """
        
        xg_shots_df = self.xg_df.copy()

        # Because event_type is always one in our data, we can use it its sum as the total number of shots each player took.
        xg_shots_df.rename(columns={'event_type': 'n_shots'}, inplace=True)

        xg_shots_df['xG_per_shot_ratio'] = xg_shots_df['expectedGoals'] / xg_shots_df['n_shots']
        xg_shots_df = xg_shots_df[xg_shots_df['n_shots']>100].sort_values(['xG_per_shot_ratio', 'trueGoals'], ascending=False).reset_index(drop=True)
        xg_shots_df['rank'] = xg_shots_df.index+1
        xg_shots_df = xg_shots_df[['rank', 'player', 'xG_per_shot_ratio', 'trueGoals', 'expectedGoals', 'difference']].head(10)

        self._plot_bar(xg_shots_df, 'xG_per_shot_ratio', 'player', "Best quality shottakers", 'xG per 1 shot', x_range=np.arange(0, 0.25, 0.02))


    def plot_bot_xg_shots_ratio(self):
        """
        Plots the bottom players based on the ratio of expected goals to the number of shots taken.
        """
        
        xg_shots_df = self.xg_df.copy()

        # Because event_type is always one in our data, we can use it its sum as the total number of shots each player took.
        xg_shots_df.rename(columns={'event_type': 'n_shots'}, inplace=True)

        xg_shots_df['xG_per_shot_ratio'] = xg_shots_df['expectedGoals'] / xg_shots_df['n_shots']
        xg_shots_df = xg_shots_df[xg_shots_df['n_shots']>100].sort_values(['xG_per_shot_ratio', 'trueGoals'], ascending=False).reset_index(drop=True)
        xg_shots_df['rank'] = xg_shots_df.index + 1
        xg_shots_df = xg_shots_df[['rank', 'player', 'xG_per_shot_ratio', 'trueGoals', 'expectedGoals', 'difference']].tail(10)

        self._plot_bar(xg_shots_df, 'xG_per_shot_ratio', 'player', "Worst quality shottakers", 'xG per 1 shot', x_range=np.arange(0, 0.1,0.01))


    def plot_best_passers(self):
        """
        Plots the best passers by xG generated
        """

        # Get the shots DataFrame because the other DataFrame is already grouped by the shot taker
        shots = self.xg_model.get_shots()

        # Retrieve the data where an assist was given
        passing = shots[shots.assist_method.isin([1,4])]

        # Group by player 2 as he is the most relevant for this plot
        passing_players = passing.groupby('player2').sum().reset_index()

        passing_players.rename(columns={'player2': 'player', 'event_type': 'n_passes', 'is_goal': 'trueGoals_created', 'expectedGoals': 'expectedGoals_created'}, inplace=True)
        passing_players = passing_players.sort_values('expectedGoals_created', ascending=False).reset_index(drop=True)
        passing_players['rank'] = passing_players.index + 1

        # Save required columns
        passing_players = passing_players[['rank', 'player', 'expectedGoals_created']].head(10)

        # Use option_context to temporarily print the columns
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(passing_players)


    def _plot_bar(self, df, x_param : str, y_param : str, title : str, x_label : str, y_label = "", x_range = np.arange(0, 1.9, 0.2)):
        """
        Generates the Barplot
        """
        sns.set_style("dark")
        fig, ax = plt.subplots(figsize=[12,5])
        ax = sns.barplot(x=abs(df[x_param]), y=df[y_param], palette='viridis', alpha=0.9)
        ax.set_xticks(x_range)
        ax.set_xlabel(xlabel=x_label, fontsize=12)
        ax.set_ylabel(ylabel=y_label)
        ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=12)
        plt.title(title, fontsize=22, fontfamily='serif')
        ax.grid(color='black', linestyle='-', linewidth=0.1, axis='x')
        plt.show()



if __name__ == "__main__":
    # Creates the plot object
    plot = ExpectedGoalsPlot()

    # Choose the plot you want to show
    # plot.plot_best_xg_goal_diff()
    # plot.plot_worst_xg_goal_diff()
    plot.plot_best_ratio()
    # plot.plot_worst_ratio()
    # plot.plot_best_passers()
    # plot.plot_top_xg_shots_ratio()
    # plot.plot_bot_xg_shots_ratio()
    # plot.plot_most_xg()