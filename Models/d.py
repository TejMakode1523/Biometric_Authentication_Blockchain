import os
from collections import defaultdict


def restructure_folder(folder_path):
    # Create a dictionary to group files by their first four digits
    grouped_files = defaultdict(list)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith((".tif", ".jpg", ".jpeg")):  # Filter for image files
            prefix = filename[:2]  # Extract the first four digits
            grouped_files[prefix].append(filename)

    # Create subfolders and move files
    user_counter = 1
    for prefix, files in grouped_files.items():
        subfolder_name = f"User_{user_counter}"
        subfolder_path = os.path.join(folder_path, subfolder_name)
        os.makedirs(
            subfolder_path, exist_ok=True
        )  # Create subfolder if it doesn't exist

        for file in files:
            source_path = os.path.join(folder_path, file)
            destination_path = os.path.join(subfolder_path, file)
            os.rename(source_path, destination_path)  # Move file to subfolder

        user_counter += 1


if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing the images: ").strip()
    if os.path.exists(folder_path):
        restructure_folder(folder_path)
        print("Folder restructuring completed successfully.")
    else:
        print("The specified folder path does not exist.")
