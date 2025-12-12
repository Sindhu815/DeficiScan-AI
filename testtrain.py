# === Simple model evaluation script ===
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# 1) Paths (EDIT IF NEEDED)
# -----------------------------
BASE_DIR = r"C:\Users\msind\OneDrive\Documents\Desktop\cnn model"
MODEL_PATH = os.path.join(BASE_DIR, "FinalMP_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "label_classes.npy")
TEST_DIR = os.path.join(BASE_DIR, "trainingimages")   # put your test folder here

# -----------------------------
# 2) Basic params (match training)
# -----------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# -----------------------------
# 3) Build test generator
# -----------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(test_gen.class_indices.keys())
print("\nDetected classes (test):", class_names)

# -----------------------------
# 4) Load saved class mapping (from training) and verify
# -----------------------------
saved_indices = np.load(LABELS_PATH, allow_pickle=True).item()
print("Saved class_indices (train):", saved_indices)
print("Test  class_indices:", test_gen.class_indices)

if saved_indices != test_gen.class_indices:
    print("\nWARNING: Class index order mismatch between training and test!")
    print("Make sure TEST_DIR has the same subfolder names and sorting as training.")
    # You can still proceed, but metrics may be wrong due to label mapping.

# -----------------------------
# 5) Load model and compile
# -----------------------------
model = load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# 6) Evaluate (accuracy & loss)
# -----------------------------
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# -----------------------------
# 7) Detailed metrics
# -----------------------------
print("\nGetting predictions...")
probs = model.predict(test_gen, verbose=0)
y_pred = probs.argmax(axis=1)
y_true = test_gen.classes

print("\nClassification report (per class):")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows=true, cols=pred):\n", cm)

# -----------------------------
# 8) Show a few sample predictions
# -----------------------------
print("\nSample predictions:")
for i in range(min(8, len(test_gen.filenames))):
    pred_idx = y_pred[i]
    true_idx = y_true[i]
    conf = probs[i][pred_idx]
    print(f"- {test_gen.filenames[i]} | True: {class_names[true_idx]} | Pred: {class_names[pred_idx]} | Conf: {conf:.3f}")

# -----------------------------
# 9) Save all predictions for review
# -----------------------------
out_df = pd.DataFrame({
    "file": test_gen.filenames,
    "true": [class_names[i] for i in y_true],
    "pred": [class_names[i] for i in y_pred],
    "confidence": probs.max(axis=1)
})
csv_path = os.path.join(BASE_DIR, "test_predictions.csv")
out_df.to_csv(csv_path, index=False)
print(f"\nSaved detailed predictions to: {csv_path}")

# -----------------------------
# 10) Optional: Quick top-2 accuracy
# -----------------------------
top2 = tf.keras.metrics.top_k_categorical_accuracy(
    tf.one_hot(y_true, depth=len(class_names)), probs, k=2
).numpy().mean()
print(f"Top-2 Accuracy (optional): {top2*100:.2f}%")
