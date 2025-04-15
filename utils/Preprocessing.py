import pandas as pd  
import numpy as np
import matplotlib.dates as mdates


def combine_activities(df_one, df_two, output_path):
    # Select data from each DataFrame based on curb_activity values
    #curb_activity: 1.0 is cross down, 0.0 is cross up, curb_type: 1.0 is curb, 2.0 is ramp, 3.0 is transition stone
    # cross curb down
    activity_one = df_one[(df_one['curb_activity'] == 1.0) &(df_one['curb_type_down'] == 1.0)]
    # cross curb up 
    # activity_two = df_two[(df_two['curb_activity'] == 0.0) & (df_two['curb_type_up'] == 1.0)]
    # cross ramp down
    activity_three = df_two[(df_two['curb_activity'] == 1.0) & (df_two['curb_type_down'] == 2.0)]
    # Combine the selected data
    df_combined = pd.concat([activity_one,activity_three], ignore_index=True)
    # Sort by timestamp and reset index
    df_combined = df_combined.sort_values('NTP').reset_index(drop=True)
    # Save the combined DataFrame
    df_combined.to_csv(output_path, index=False)
    return df_combined

    
def handle_missing_values_length_index(df_filtered):
    # Dictionary to store consecutive missing info (start_index: count_of_missing)
    missing_info = {}
    current_count = 0
    missing_value_flag = False
    # Will store the first index of a missing-values sequence (None means we're currently not in a missing sequence)
    start_index = None

    # Iterate over each row in df_filtered
    for index, row in df_filtered.iterrows():
        # Check if 'Acc-Z' is NaN
        if np.isnan(row['Acc-Z']):
            # If this is the first missing in a new sequence, record the start index
            if start_index is None:
                start_index = index
                # Start counting from 1 for this missing value
                current_count += 1
            else:
                # If we're already tracking a missing sequence, just increment
                current_count += 1
        else:
            # If 'Acc-Z' is not NaN so that the missing sequence broke
            if start_index is not None:
                # Store start_index -> length_of_missing_sequence in the dictionary
                missing_info[start_index] = current_count
                # Reset to indicate we're no longer in a missing sequence
                start_index = None
                current_count = 0
    # Handle case where the last samples in df_filtered are missing
    # (Because if the loop ends while still in a missing sequence, it's not recorded yet)
    if current_count > 0:
        missing_info[start_index] = current_count
    # Print sequences longer than 100 missing values (twice for demonstration)
    for key, value in missing_info.items():
        if value > 100:
            print("Start Index:", key, "Count:", value)
            missing_value_flag = True
    # If no sequences are longer than 100, print a message
    if not missing_value_flag:
        print("No missing values longer than 100")
    return missing_info


#not working because NTP IS NAN 
def handle_missing_values_time_diff(df_filtered, time_threshold=0.5):
    missing_info = {}               # {start_time: time_diff}
    missing_value_flag = False      # Tracks if any sequence exceeded threshold
    start_time = None     # Time when we first encounter a missing value
    last_time = None      # Most recent time in the current missing sequence
    # Sort the DataFrame by time just in case
    df_filtered = df_filtered.sort_values('NTP')

    for index, row in df_filtered.iterrows():
        if np.isnan(row['Acc-Z']):
            # Start a new missing sequence if we're not already in one
            if start_time is None:
                start_time = index
                print(start_time)
            #start_time not None, so we're in the middle of a missing sequence
            else:
                last_time = index
        # If 'Acc-Z' is not NaN so that the missing sequence broke
        else:
            if start_time is not None and last_time is not None:
                time_diff = last_time - start_time
                missing_info[start_time] = time_diff
                # Reset for the next missing sequence
                start_time = None
                last_time = None

    # After iterating, if we ended in the middle of a missing sequence
    if start_time is not None and last_time is not None:
        time_diff = last_time - start_time
        missing_info[start_time] = time_diff

    # # Print sequences that exceed the threshold
    # for start_t, diff in missing_info.items():
    #      if diff > time_threshold:
    #         print("Start time:", start_t, "Dauer:", diff)
    #         missing_value_flag = True

    # if not missing_value_flag:
    #     print(f"No missing values longer than {time_threshold} second(s).")

    return missing_info


# Function to fill missing values based on the specified conditions
def fill_missing_values(df,output_path):
    df['NTP'] = pd.to_datetime(df['NTP'])
    for index in range(1, len(df)):
        if pd.isnull(df['Acc-Z'].iloc[index]):
            # If the previous value is not NaN, use that value
            if df['curb_scene'].iloc[index - 1] == df['curb_scene'].iloc[index]:
                df.at[index, 'Acc-Z'] = df['Acc-Z'].iloc[index - 1]
                df.at[index, 'NTP'] = df['NTP'].iloc[index - 1] + pd.Timedelta(milliseconds=1)
            else:
                # Find the next non-NaN value below
                for j in range(index + 1, len(df)):
                    if not pd.isnull(df['Acc-Z'].iloc[j]):
                        df.at[index, 'Acc-Z'] = df['Acc-Z'].iloc[j]
                        df.at[index, 'NTP'] = df['NTP'].iloc[j] - pd.Timedelta(milliseconds=1)
                        break
    df.to_csv(output_path, index=False)

#from Dandan and no idea what it does
def down_sampling(data, frequency):
    time_shift = 1 / frequency  # s
    # exchange time shift to unix time in ms
    time_shift = time_shift*1000
    # print('time interval',time_shift)
    sampled_data = []

    ts = 'Timestamp_unix'
    # print('data',data)
    sampled_data.append(data.iloc[0])
    last_source_data = data.iloc[0]
    nex_timestamp = data[ts][0] + time_shift
    # print(data[ts][0])

    for i in range(1, data.shape[0]):
        # print('current timestamp', data['Timestamp_unix'][i])
        # print(data['Timestamp_unix'][i] - last_source_data['Timestamp_unix'])
        # if the gap between two consecutive time is larger than 2s, take the next sample
        if (data[ts][i] - last_source_data[ts]) > 2000.0:
            # print('current timestamp', data['Timestamp_unix'][i])
            # print('last source timestamp', last_source_data['Timestamp_unix'])
            nex_timestamp = data[ts][i] + time_shift
            # append the current data to sampled data
            sampled_data.append(data.iloc[i,])

        if data[ts][i] > nex_timestamp:
            nex_timestamp += time_shift
            # append the current data to sampled data
            sampled_data.append(data.iloc[i,])
        last_source_data = data.iloc[i,]

    sampled_data = pd.DataFrame(np.array(sampled_data), columns=(data.columns))
    # print('time interval median',sampled_data[[ts]].diff())
    # print('after sampling',sampled_data.shape)

    return sampled_data

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
    df.loc[mask_ramp_down, 'curb_height'] = df.loc[mask_ramp_down, 'curb_height_down']
    # Fill any missing values with 0.0
    if df['curb_height'].isnull().any():
        df['curb_height'] = df['curb_height'].fillna(0.0)
    # Filter and return relevant columns
    df_filtered = df[['NTP','Acc-Z', 'curb_scene', 'curb_height']]
    return df_filtered


