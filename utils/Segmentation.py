import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates


def segment_acceleration_data_no_overlapping(df,output):
    df.loc[:, 'NTP'] = pd.to_datetime(df['NTP'])
    processed_segments = []
    grouped = df.groupby('curb_scene')
    for name, group in grouped:
        group = group.sort_values(by='NTP')
        group.set_index('NTP', inplace=True)
        resampled = group.resample('0.5s')
        for index, segment in resampled:
            if len(segment) < 80:
                continue
            acc_z_values = segment['Acc-Z'].values
            curb_scene_value = segment['curb_scene'].iloc[0]
            start_time = segment.index[0]
            end_time = segment.index[-1]
            data = {'curb_scene': curb_scene_value, 'start_time': start_time, 'end_time': end_time}
            
            # Ensure acc_z_values has exactly 100 data points
            if len(acc_z_values) > 100:
                acc_z_values = acc_z_values[:100]
            elif len(acc_z_values) < 100:
                acc_z_values = list(acc_z_values) + [acc_z_values[-1]] * (100 - len(acc_z_values))

            for j, value in enumerate(acc_z_values):
                data[f'Acc-Z_{j+1}'] = value
            new_df = pd.DataFrame([data])
            processed_segments.append(new_df)
        final_df = pd.concat(processed_segments, ignore_index=True)
        # Save the final DataFrame to a CSV file
        final_df.to_csv(output, index=False)
    return final_df