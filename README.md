# 🌦️ Weather Classification with Decision Trees

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end computer vision and machine learning pipeline that classifies outdoor weather conditions into four distinct categories using **OpenCV** image preprocessing and a **Scikit-Learn Decision Tree Classifier**.

---

## 📌 Project Overview

Weather condition recognition is a fundamental task for outdoor autonomous systems, smart photography, and environmental monitoring. This project explores a baseline **Decision Tree Classifier** trained on flattened RGB pixel intensity features to distinguish between four atmospheric conditions:

* ☁️ **Cloudy**
* 🌧️ **Rain**
* ☀️ **Shine / Sunshine**
* 🌅 **Sunrise**

---

## 🚀 Methodology & Pipeline

```
[Raw Outdoor Image] ➔ [RGB Conversion & Resize (128x128)] ➔ [Normalize & Flatten (49,152 features)] ➔ [Decision Tree Classifier] ➔ [Weather Prediction]
```

1. **Image Preprocessing (`OpenCV` & `NumPy`)**:
   * **Color Space Conversion**: Converted from default OpenCV BGR to standard **RGB**.
   * **Spatial Resizing**: All images standardized to **128 × 128 pixels**.
   * **Intensity Normalization**: Pixel values scaled from `[0, 255]` to `[0.0, 1.0]`.
   * **Feature Flattening**: Each 3D image array (`128, 128, 3`) is flattened into a 1D feature vector of **49,152 dimensions**.

2. **Model Architecture (`Scikit-Learn`)**:
   * Algorithm: **Decision Tree Classifier** (`criterion='gini'`, `random_state=42`)
   * Trained on **937 training images** and evaluated on **188 unseen test images**.

---

## 📊 Performance & Results

The model achieved an overall test accuracy of **69.68%**. Below is the detailed class-level evaluation on the test dataset:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| ☁️ **Cloudy** | `0.55` | `0.56` | `0.55` | 50 |
| 🌧️ **Rain** | `0.68` | `0.64` | `0.66` | 47 |
| ☀️ **Shine** | `0.87` | `0.72` | `0.79` | 47 |
| 🌅 **Sunrise** | `0.72` | `0.89` | `0.80` | 44 |
| **Overall Accuracy** | — | — | **`69.68%`** | **188** |
| **Weighted Average** | `0.70` | `0.70` | `0.70` | 188 |

> **Key Observation**: The classifier achieves its highest F1-scores on **Sunrise** (`0.80`) and **Shine** (`0.79`), where distinct warm color distributions (oranges/yellows) provide strong separating hyperplanes in pixel feature space.
dataset- https://drive.google.com/drive/folders/1ZOeIu-rv8Env7u7nqaf1K45htnMBkSs3?usp=sharing
## 📂 Repository Structure

```text
Weather_Classification/
│
├── .gitignore                      
├── README.md                       
├── Weather_Classification.ipynb    
├── sunny.jpg                       
├── test2.jpg                       
│
└── dataset/                        
     ├── train/
     │    ├── cloudy/
     │    ├── rain/
     │    ├── shine/
     │    └── sunrise/
     └── test/
          ├── cloudy/
          ├── rain/
          ├── shine/
          └── sunrise/

## ⚙️ Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/abelm10/Weather_Classification.git
cd Weather_Classification
```

### 2. Install Dependencies
```bash
pip install numpy opencv-python scikit-learn matplotlib
```

### 3. Setup the Dataset
To keep repository size lightweight, the image dataset is ignored by `.gitignore`. 
* Download your weather image dataset and place the `train/` and `test/` folders inside a `dataset/` directory at the project root matching the folder structure shown above.

### 4. Run the Pipeline
Open the Jupyter Notebook in VS Code or Jupyter Lab:
```bash
jupyter notebook Weather_Classification.ipynb
```
Execute the cells sequentially to load the dataset, train the classifier, generate the confusion matrix, and run inference on custom sample images like `test2.jpg`.

---

## 🔮 Future Improvements
* **Feature Engineering**: Extract color histograms and spatial edge features (e.g., HOG, Sobel) instead of raw pixel values.
* **Deep Learning Baseline**: Implement a Convolutional Neural Network (CNN) or transfer learning with MobileNetV2 to capture spatial hierarchies and improve accuracy beyond ~70%.
* **Hyperparameter Tuning**: Optimize tree depth (`max_depth`) and minimum samples per leaf (`min_samples_leaf`) to prevent overfitting.

*GitHub: [@abelm10](https://github.com/abelm10)*
README.md
Displaying README.md.
