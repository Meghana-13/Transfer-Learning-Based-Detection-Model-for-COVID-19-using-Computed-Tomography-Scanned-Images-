# Transfer-Learning-Based-Detection-Model-for-COVID-19-using-Computed-Tomography-Scanned-Images

# Transfer Learning Based Detection Model for COVID-19 using Computed Tomography (CT) Scanned Images

This repository presents a deep transfer learning-based approach for the detection of COVID-19 using CT scan images. The model compares the performance of multiple pre-trained convolutional neural networks (CNNs), including ResNet50, VGG16, and VGG19, to evaluate their effectiveness in identifying COVID-19 cases from chest CT scans.

📌 Overview

The aim of this project is to develop an accurate and robust model that leverages transfer learning to classify CT images into COVID-19 and non-COVID categories. The models were trained and validated using the [SARS-CoV-2 CT-scan dataset](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset) available on Kaggle.

📂 Dataset

- Source: Kaggle SARS-CoV-2 CT-scan Dataset
- Total Images: ~2482 images
  - COVID-19: 1252 images
  - Non-COVID: 1230 images
- Format: JPEG
- Annotations: Binary class labels (COVID, Non-COVID)

🧠 Models Used

We utilized the following transfer learning models with fine-tuning:

- ✅ ResNet50
- ✅ VGG16
- ✅ VGG19

All models are initialized with pre-trained ImageNet weights.

🛠️ Tools and Technologies

- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- OpenCV
- Google Colab / Jupyter Notebook

📊 Performance Metrics

We evaluated the models using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix


