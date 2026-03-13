import os

# Let's see where we are
print(f"Current Working Directory: {os.getcwd()}")

# Check if the folders exist
folders_to_check = ["data/raw", "data/empyema"]

for folder in folders_to_check:
    if os.path.exists(folder):
        print(f"✅ Folder '{folder}' found!")
        # Let's see what's inside
        subfolders = os.listdir(folder)
        print(f"   Contains: {subfolders[:5]}...") # show first 5 items
    else:
        print(f"❌ Folder '{folder}' NOT found. Check your spelling or folder location.")

# Check for a specific image extension
# Sometimes Kaggle images are .JPG (uppercase) and the script looks for .jpg (lowercase)