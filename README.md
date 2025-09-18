Nail & Hair Deficiency Detection (CNN)
CNN-based image classifier that analyzes nail and hair photos to flag potential nutrient deficiencies for learning and demonstration purposes. Built with TensorFlow/Keras and Python, includes training and single-image inference.

-----Features
End-to-end pipeline: training, testing, and single-image prediction.

Simple CLI scripts: trainingcode.py, testtrain.py, test.py.

CSV output of predictions for easy sharing.

-----Repository layout
trainingcode.py — train a CNN on prepared images.

testtrain.py — evaluate on a test split.

test.py — run prediction for one image; writes test_predictions.csv or single_image_test_predictions.csv.

.gitignore — excludes datasets, models, caches.

Quick start
----Prerequisites

Python 3.10

pip install -r requirements.txt
If requirements.txt isn’t present, install the basics:

pip install tensorflow keras numpy opencv-python matplotlib

-----Prepare data and model

Datasets (rawdata, trainingimages, test_data) and trained model (FinalMP_model.h5) are not stored in Git to keep the repo small. Download or collect images and place them in a data/ folder; put a trained model at model/FinalMP_model.h5 if using pretrained weights.

----Train

python trainingcode.py
Edit paths inside the script if required.

-----Evaluate

python testtrain.py
Shows metrics on the test split.

------Predict a single image

python test.py --image path/to/image.jpg
Creates a CSV (test_predictions.csv or single_image_test_predictions.csv) with predicted class and confidence.
------Notes
Educational project; not medical advice.