import os
import shutil

txt_folder = ''
image_folder = ''
filtered_image_folder = ''

os.makedirs(filtered_image_folder, exist_ok=True)

txt_file_names = [os.path.splitext(filename)[0] for filename in os.listdir(txt_folder) if filename.endswith('.txt')]

for image_filename in os.listdir(image_folder):
    if image_filename.endswith('.jpg'):
        image_name = os.path.splitext(image_filename)[0]
        if image_name in txt_file_names:
            src_path = os.path.join(image_folder, image_filename)
            dst_path = os.path.join(filtered_image_folder, image_filename)
            shutil.copy(src_path, dst_path)

print(f"Filtered images have been copied to {filtered_image_folder}")
