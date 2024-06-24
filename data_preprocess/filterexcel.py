import os
import pandas as pd

image_folder = '/'
excel_file_path = '/'
output_excel_path = ''

image_names = [os.path.splitext(filename)[0] for filename in os.listdir(image_folder)]

df = pd.read_excel(excel_file_path)

filtered_df = df[df['ImageID'].isin(image_names)]

filtered_df = filtered_df[filtered_df['LabelName'] == '/m/04dr76w']

filtered_df.to_excel(output_excel_path, index=False)

print(f"Filtered data has been saved to {output_excel_path}")