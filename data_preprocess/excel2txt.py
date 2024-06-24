import pandas as pd
import os

excel_file_path = ''
output_directory = ''

os.makedirs(output_directory, exist_ok=True)

df = pd.read_excel(excel_file_path)

print("Columns in the Excel file:", df.columns)

if 'ImageID' in df.columns:
    for image_id in df['ImageID'].unique():
        rows = df[df['ImageID'] == image_id]

        txt_content = ""
        for index, row in rows.iterrows():
            values = row.iloc[1:6].astype(str).tolist() 
            txt_content += " ".join(values) + "\n"
      
        txt_file_path = os.path.join(output_directory, f"{image_id}.txt")
        with open(txt_file_path, 'w') as txt_file:
            txt_file.write(txt_content)

    print(f"Text files have been created in {output_directory}")
else:
    print("The column 'ImageID' was not found in the Excel file.")
