import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates



def combine_activities(df_one, df_two, output_path):
    """
    Combine two DataFrames based on specified curb_activity values and save the result.
    
    Parameters:
    -----------
    df_one : pandas.DataFrame
        First DataFrame containing curb_activity
    df_two : pandas.DataFrame
        Second DataFrame containing curb_activity
    output_path : str, default='../data/handlebar_processed.csv'
        Path where the combined DataFrame will be saved
    Returns:
    --------
    pandas.DataFrame
        Combined and sorted DataFrame
    """
    # Select data from each DataFrame based on curb_activity values
    activity_one = df_one[df_one['curb_activity'] == 1.0]
    activity_two = df_two[df_two['curb_activity'] == 2.0]
    # Combine the selected data
    df_combined = pd.concat([activity_one, activity_two], ignore_index=True)
    # Sort by timestamp and reset index
    df_combined = df_combined.sort_values('NTP').reset_index(drop=True)
    # Save the combined DataFrame
    df_combined.to_csv(output_path, index=False)
    
    return df_combined