import os
import cv2
import numpy as np
from tqdm import tqdm # Install this: pip install tqdm (it shows a progress bar)

RAW_DATA_DIR = "data/raw"
EMPYEMA_DIR = "data/empyema"
PROCESSED_DIR = "data/processed"
IMG_SIZE = 224

def augment_image(img):
    """Creates variations of an image to help with small datasets."""
    aug_images = []
    aug_images.append(cv2.flip(img, 1)) # Horizontal Flip
    
    # Rotate slightly
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1) # 15 degree rotation
    aug_images.append(cv2.warpAffine(img, M, (cols, rows)))
    
    # Brightness adjustment
    aug_images.append(cv2.convertScaleAbs(img, alpha=1.2, beta=10)) 
    
    return aug_images

def process_and_save():
    # 1. Clean up and create folders
    categories = ['pituitary', 'no_tumor', 'meningioma', 'glioma', 'empyema']
    for cat in categories:
        os.makedirs(os.path.join(PROCESSED_DIR, cat), exist_ok=True)

    print("🚀 Starting Preprocessing...")

    # 2. Process the Kaggle Folders (Training & Testing)
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                
                # Figure out the category from the folder name
                category = "no_tumor"
                if "pituitary" in root.lower(): category = "pituitary"
                elif "meningioma" in root.lower(): category = "meningioma"
                elif "glioma" in root.lower(): category = "glioma"
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    save_name = f"{category}_{file.split('.')[0]}.npy"
                    np.save(os.path.join(PROCESSED_DIR, category, save_name), img/255.0)

    # 3. Process & AUGMENT Empyema (The special part)
    print("✨ Augmenting Empyema samples...")
    empyema_files = [f for f in os.listdir(EMPYEMA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for i, file in enumerate(empyema_files):
        img_path = os.path.join(EMPYEMA_DIR, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Save original
            np.save(os.path.join(PROCESSED_DIR, 'empyema', f"emp_orig_{i}.npy"), img/255.0)
            
            # Save 10 augmented versions for every 1 original image
            for j in range(10): 
                # Create a variation
                M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), np.random.randint(-20, 20), 1)
                aug_img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
                if np.random.rand() > 0.5: aug_img = cv2.flip(aug_img, 1)
                
                np.save(os.path.join(PROCESSED_DIR, 'empyema', f"emp_aug_{i}_{j}.npy"), aug_img/255.0)

    print(f"✅ Finished! Check {PROCESSED_DIR}")

if __name__ == "__main__":
    process_and_save()