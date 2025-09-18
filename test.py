import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# === Paths ===
BASE_DIR = r"C:\Users\msind\OneDrive\Documents\Desktop\cnn model"
MODEL_PATH = os.path.join(BASE_DIR, "FinalMP_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "label_classes.npy")
TEST_DIR = os.path.join(BASE_DIR, "test_data")

# === Image parameters ===
IMG_HEIGHT = 224
IMG_WIDTH = 224

# === Load model & labels ===
print("Loading model...")
model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True).item()
class_labels = [cls for cls, idx in sorted(labels.items(), key=lambda x: x[1])]

# === Get test images ===
test_images = []
for root, _, files in os.walk(TEST_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(os.path.join(root, f))

if not test_images:
    print("No images found!")
    exit()

print(f"Found {len(test_images)} images\n")

# === Show results in smaller, cleaner grid ===
num_images = len(test_images)
cols = 5  # More columns for smaller images
rows = (num_images + cols - 1) // cols

# Create figure with smaller sizing
fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows))
fig.suptitle('Deficiency Detection Results', fontsize=14, fontweight='bold')

# Handle single row case
if rows == 1:
    axes = axes.reshape(1, -1) if num_images > 1 else [axes]
elif num_images == 1:
    axes = np.array([[axes]])

for i in range(rows * cols):
    row_idx = i // cols
    col_idx = i % cols
    
    if i < num_images:
        try:
            # Load and preprocess image
            img = load_img(test_images[i], target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_array, verbose=0)[0]
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Display
            if rows == 1:
                ax = axes[col_idx] if num_images > 1 else axes
            else:
                ax = axes[row_idx, col_idx]
                
            ax.imshow(img)
            ax.axis('off')
            
            # Clean up the prediction name
            display_name = predicted_class.replace('_', ' ')
            if ',' in display_name:
                display_name = 'Mineral\nDeficiency'  # Simplify long name
            
            # Simple color coding - ONLY for text color
            if confidence > 75:
                color = 'black'  # High confidence = black text
            elif confidence > 50:
                color = 'darkorange'  # Medium confidence = orange text
            else:
                color = 'red'  # Low confidence = red text
            
            title = f"{display_name}\n{confidence:.0f}%"
            ax.set_title(title, fontsize=9, fontweight='bold', color=color, pad=8)
            
        except Exception as e:
            if rows == 1:
                ax = axes[col_idx] if num_images > 1 else axes
            else:
                ax = axes[row_idx, col_idx]
            ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    else:
        # Hide empty subplots
        if rows == 1:
            ax = axes[col_idx] if num_images > 1 else axes
        else:
            ax = axes[row_idx, col_idx]
        ax.axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("WHAT DO THE COLORS MEAN?")
print("="*50)
print("🖤 BLACK text  = High confidence (>75%) - Trust this result")
print("🟠 ORANGE text = Medium confidence (50-75%) - Somewhat unsure") 
print("🔴 RED text    = Low confidence (<50%) - Very unsure, ignore")
print()
print("PREDICTION RESULTS:")
print("-" * 50)

for i, img_path in enumerate(test_images, 1):
    try:
        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)[0]
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        display_name = predicted_class.replace('_', ' ')
        if ',' in display_name:
            display_name = 'Mineral Deficiency'
            
        confidence_level = "HIGH" if confidence > 75 else "MED" if confidence > 50 else "LOW"
        print(f"{i:2d}. {display_name:20} - {confidence:5.1f}% ({confidence_level})")
        
    except:
        print(f"{i:2d}. Error processing image")
