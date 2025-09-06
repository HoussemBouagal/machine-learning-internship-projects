# 🔥 Machine Learning Internship Projects 

This repository contains **advanced Machine Learning tasks** completed as part of my internship at **Codveda Technologies**.  
Each task demonstrates expertise in **data preprocessing, model building, hyperparameter tuning, class balancing, evaluation, and visualization**.  
All projects are reproducible with saved models and assets.

---

## 📂 Repository Structure
```
machine-learning-internship-projects/
├── Task1/                      # 🔹 Random Forest - Customer Churn Prediction
│   ├── assets/                 # Visualizations: feature importance, learning curve, confusion matrix
│   ├── model/                  # Trained Random Forest model, label encoders, and scaler
│   ├── requirements.txt        # Python dependencies for Task 1
│   ├── randomforest-churn-task1.ipynb # Full implementation for churn prediction
│   └── README.md               # Documentation for Task 1 (approach, results, insights)
│
├── Task2/                      # 🔹 SVM Classification - Iris Binary Classification (Versicolor vs Virginica)
│   ├── linear_results/         # Results & plots for Linear Kernel SVM
│   ├── rbf_results/            # Results & plots for RBF Kernel SVM
│   ├── model/                  # Trained SVM models and scaler
│   ├── requirements.txt        # Python dependencies for Task 2
│   ├── svm-iris-classification.ipynb # Notebook with SVM training, evaluation & visualization
│   └── README.md               # Documentation for Task 2 (methods, results, kernel comparison)
│
├── Task3/                      # 🔹 Neural Networks - Iris Multi-Class Classification (Setosa, Versicolor, Virginica)
│   ├── model/                  # Best Keras model saved during training
│   ├── assets/                 # Confusion matrix & training metrics plots
│   ├── requirements.txt        # Python dependencies for Task 3
│   ├── iris-deeplearning-analysis.ipynb # Notebook with neural network design & evaluation
│   └── README.md               # Documentation for Task 3 (NN architecture, results, insights)
│
└── README.md                   # 📜 Main documentation (project overview, structure, how to run)
```

---

## 🚀 Tasks Overview

### 🔹 [Task 1: Random Forest - Customer Churn Prediction](Task1/README.md)
- **Goal:** Predict **customer churn** using a **Random Forest Classifier**.
- **Techniques:** SMOTE balancing, GridSearchCV tuning, Cross-Validation.
- **Highlights:**
  - Train Accuracy: **96.07%**
  - Test Accuracy: **94.30%**
- **Visualizations:** Feature Importance, Confusion Matrix, Learning Curve.

---

### 🔹 [Task 2: SVM Classification - Iris Binary Classification](Task2/README.md)
- **Goal:** Classify **Versicolor vs Virginica** in the Iris dataset using **SVM**.
- **Techniques:** Linear & RBF kernels, SMOTE balancing, Decision Boundary visualization.
- **Highlights:**
- **SVM Linear**	
  - Test Accuracy: **90%**
  - AUC (Linear): **1.0000**
   - **SVM RBF**
  - Test Accuracy: **90%**
  - AUC (Linear): **0.9700**

- **Visualizations:** Decision Boundaries, Confusion Matrices.

---

### 🔹 [Task 3: Neural Networks - Iris Multi-Class Classification](Task3/README.md)
- **Goal:** Classify **3 Iris flower species** using a deep neural network built with **TensorFlow/Keras**.
- **Techniques:** SMOTE balancing, EarlyStopping, ModelCheckpoint, Metrics Visualization.
- **Highlights:**
  - Train Accuracy: **97.92%**
  - Validation Accuracy: **95.83%**
  - Test Accuracy: **93.33%**
- **Visualizations:** Confusion Matrix, Accuracy/Loss Curves.

---

## 🔧 Requirements
Install all required dependencies:
```bash
pip install -r requirements.txt
```
*(Each task also contains its own `requirements.txt` file.)*

---

## 🏃‍♂️ How to Run
1. Clone the repository:
```bash
git clone https://github.com/HoussemBouagal/machine-learning-internship-projects.git
cd machine-learning-internship-projects
```
2. Open the desired task directory (`Task1/`, `Task2/`, `Task3/`).
3. Install dependencies and run Jupyter notebooks:
```bash
pip install -r requirements.txt
jupyter notebook
```

---

## 🏆 Author
**Houssem Eddine Bouagal**  
*Machine Learning Intern @ Codveda Technologies*
