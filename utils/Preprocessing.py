import pandas as pd  
import numpy as np
import matplotlib.dates as mdates

def combine_activities(df_one, df_two, output_path):
    """
    Combines and processes specific curb crossing activities from two dataframes.
    
    Args:
        df_one (pd.DataFrame): First dataframe containing curb crossing data
        df_two (pd.DataFrame): Second dataframe containing curb crossing data
        output_path (str): Path where the combined CSV file will be saved
    
    Activity Types:
        - curb_activity: 1.0 = crossing down, 0.0 = crossing up
        - curb_type: 1.0 = curb, 2.0 = ramp, 3.0 = transition stone
    
    Returns:
        pd.DataFrame: Combined and sorted dataframe containing selected crossing activities
    """
    # Extract curb-down activities (crossing curb downwards)
    activity_one = df_one[(df_one['curb_activity'] == 1.0) & 
                         (df_one['curb_type_down'] == 1.0)]
    
    # Extract ramp-down activities (crossing ramp downwards)
    activity_three = df_two[(df_two['curb_activity'] == 1.0) & 
                          (df_two['curb_type_down'] == 2.0)]
    
    # Note: curb-up activities are currently commented out because they are not very correct
    # activity_two = df_two[(df_two['curb_activity'] == 0.0) & 
    #                      (df_two['curb_type_up'] == 1.0)]
    
    # Combine selected activities and reset the index
    df_combined = pd.concat([activity_one, activity_three], ignore_index=True)
    
    # Sort the combined data by timestamp (NTP)
    df_combined = df_combined.sort_values('NTP').reset_index(drop=True)
    
    # Save the processed data to CSV
    df_combined.to_csv(output_path, index=False)
    
    return df_combined


def handle_missing_values_length_index(df_filtered):
    """
    Analyzes sequences of missing values in the Acc-Z column and tracks their indices.

    This function identifies consecutive sequences of NaN values in the Acc-Z column,
    records their starting indices and lengths, and reports sequences longer than 100 values.
    
    Args:
        df_filtered (pd.DataFrame): DataFrame containing Acc-Z measurements
        
    Returns:
        dict: Dictionary with start indices as keys and sequence lengths as values
              Format: {start_index: length_of_sequence}
              
    Example:
        If there are 150 consecutive NaN values starting at index 1000,
        the function will print:
        "Start Index: 1000 Count: 150"
    """
    # Initialize tracking variables
    missing_info = {}  # Dictionary to store {start_index: sequence_length}
    current_count = 0  # Counter for current sequence length
    missing_value_flag = False  # Flag to track if any long sequences were found
    start_index = None  # Tracks start of current missing sequence
    
    # Analyze each row in the DataFrame
    for index, row in df_filtered.iterrows():
        if np.isnan(row['Acc-Z']):
            if start_index is None:
                # Start of a new missing sequence
                start_index = index
                current_count = 1
            else:
                # Continue counting current sequence
                current_count += 1
        else:
            # End of a missing sequence (if any)
            if start_index is not None:
                missing_info[start_index] = current_count
                start_index = None
                current_count = 0
    
    # Handle case where DataFrame ends with missing values
    if current_count > 0:
        missing_info[start_index] = current_count
    
    # Report sequences longer than 100 values
    for key, value in missing_info.items():
        if value > 100:
            print("Start Index:", key, "Count:", value)
            missing_value_flag = True
            
    if not missing_value_flag:
        print("No missing values longer than 100")
        
    return missing_info

def fill_missing_values(df, output_path):
    """
    Fill missing values in Acc-Z column using temporal interpolation strategy.
    
    This function handles missing accelerometer Z-axis values by:
    1. Using the previous value if within the same curb scene
    2. Using the next available value if at a scene boundary
    Also updates timestamps (NTP) to maintain temporal consistency.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Acc-Z', 'NTP', and 'curb_scene' columns
        output_path (str): Path where the processed DataFrame will be saved as CSV
        
    Side Effects:
        - Modifies the input DataFrame in-place
        - Saves the processed DataFrame to a CSV file
        
    Note:
        NTP timestamps are adjusted by ±1 millisecond to maintain sequence order
    """
    # Convert NTP column to datetime format for temporal operations
    df['NTP'] = pd.to_datetime(df['NTP'])
    
    # Iterate through the DataFrame (starting from index 1)
    for index in range(1, len(df)):
        if pd.isnull(df['Acc-Z'].iloc[index]):
            # Case 1: Missing value within same curb scene
            if df['curb_scene'].iloc[index - 1] == df['curb_scene'].iloc[index]:
                # Use previous value and increment timestamp
                df.at[index, 'Acc-Z'] = df['Acc-Z'].iloc[index - 1]
                df.at[index, 'NTP'] = df['NTP'].iloc[index - 1] + pd.Timedelta(milliseconds=1)
            else:
                # Case 2: Missing value at scene boundary
                # Search forward for next valid value
                for j in range(index + 1, len(df)):
                    if not pd.isnull(df['Acc-Z'].iloc[j]):
                        df.at[index, 'Acc-Z'] = df['Acc-Z'].iloc[j]
                        df.at[index, 'NTP'] = df['NTP'].iloc[j] - pd.Timedelta(milliseconds=1)
                        break
    
    # Save processed DataFrame to CSV
    df.to_csv(output_path, index=False)


def process_curb_height(df):
    """
    Process curb height data and create a new column with height values based on conditions.
    Args:
        df (pd.DataFrame): Input DataFrame containing curb activity data
        
    Returns:
        pd.DataFrame: DataFrame with processed curb height data and filtered columns
    """
    # Create a new column for curb height
    df['curb_height'] = 0.0  # Initialize with default value
    # Situation 1: cross curb down
    mask_curb_down = (df['curb_scene'] == 1.0) & (df['curb_activity'] == 1.0) & (df['curb_type_down'] == 1.0)
    df.loc[mask_curb_down, 'curb_height'] = df.loc[mask_curb_down, 'curb_height_down']
    # Situation 2: cross ramp down
    mask_ramp_down = (df['curb_scene'] == 1.0) & (df['curb_activity'] == 1.0) & (df['curb_type_down'] == 2.0)
    df.loc[mask_ramp_down, 'curb_height'] = 4.0
    # Fill any missing values with 0.0
    if df['curb_height'].isnull().any():
        df['curb_height'] = df['curb_height'].fillna(0.0)
    # Filter and return relevant columns
    df_filtered = df[['NTP','Acc-Z', 'curb_scene', 'curb_height']]
    return df_filtered

def label_curb_scenes(df_data, df_curb, window_size=100):
    """
    Labels curb scenes in the data and extends labels to surrounding data points for validation test.
    
    Args:
        df_data (pd.DataFrame): Original dataframe containing NTP and Acc-Z data
        df_curb (pd.DataFrame): Dataframe containing curb timestamps
        window_size (int): Number of data points to label before and after each curb scene
        
    Returns:
        pd.DataFrame: DataFrame with labeled curb scenes
    """
    # Create copy with selected columns
    df_selected = df_data[['NTP', 'Acc-Z']].copy()
    df_selected['curb_scene'] = 0
    
    # Convert NTP columns to datetime
    df_selected['NTP'] = pd.to_datetime(df_selected['NTP'])
    df_curb['NTP'] = pd.to_datetime(df_curb['NTP'])
    
    # Find matching timestamps and label curb scenes
    df_selected.loc[df_selected['NTP'].isin(df_curb['NTP']), 'curb_scene'] = 1
    
    # Get indices where curb_scene is 1
    curb_indices = df_selected.index[df_selected['curb_scene'] == 1].tolist()
    
    # Extend labels to surrounding data points
    for idx in curb_indices:
        start_idx = max(0, idx - window_size)
        end_idx = min(len(df_selected) - 1, idx + window_size)
        df_selected.loc[start_idx:end_idx, 'curb_scene'] = 1
    
    return df_selected


