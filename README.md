# Deep Learning PR 1 — Breast Cancer Classification with Neural Networks

## Project Description

This project builds and compares several neural network models to classify breast tumor samples
as malignant or benign, using the Breast Cancer Wisconsin (Diagnostic) dataset. The notebook covers
data exploration and preprocessing, a single-layer perceptron baseline, a multi-layer perceptron
with different activation functions, early stopping, dropout regularization, and weight
regularization (L1, L2, L1-L2). It ends with a full comparison of all models and a discussion of
which model is best suited for a clinical decision-support tool.

## Dataset Information

- Source: `sklearn.datasets.load_breast_cancer(as_frame=True)`
- Samples: 569
- Features: 30 numeric features
- Target: binary (0 = Malignant, 1 = Benign)
- Class distribution: 212 malignant, 357 benign (mild class imbalance)
- No missing values

## Technologies Used

- Python
- TensorFlow / Keras
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- Jupyter Notebook

## Tasks Covered

1. Data Loading, EDA & Preprocessing
2. Single-Layer Perceptron (SLP) — Baseline Model
3. Multi-Layer Perceptron (MLP) — Activation Functions
4. Early Stopping
5. Dropout Layers
6. Regularization (L1, L2, L1-L2)
7. Final Comparison, Business Insights & Notebook Quality

## Models Used

- Single-Layer Perceptron (SLP): one Dense(1, sigmoid) neuron
- Multi-Layer Perceptron (MLP): Dense(64) → Dense(32) → Dense(1), compared with ReLU, Tanh and
  Sigmoid hidden-layer activations
- MLP with Early Stopping: Dense(128) → Dense(64) → Dense(1)
- MLP with Dropout: Dense(128) → Dropout → Dense(64) → Dropout → Dense(1), compared at dropout
  rates 0.1, 0.3 and 0.5
- MLP with L1, L2 and L1-L2 regularization
- Final Combined Model: Dense(128, L2) → Dropout(0.3) → Dense(64, L2) → Dropout(0.3) → Dense(1),
  trained with Early Stopping

## Key Results

Results from the actual notebook run (`Restart & Run All`, zero errors):

| Model                     | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
|---------------------------|:---:|:---:|:---:|:---:|
| SLP                       | 0.9211 | 0.9437 | 0.9306 | 0.9371 |
| MLP - ReLU (best activation) | 0.9737 | 0.9859 | 0.9722 | 0.9790 |
| MLP + Early Stopping      | 0.9649 | 0.9857 | 0.9583 | 0.9718 |
| MLP + Dropout (best rate 0.1) | 0.9561 | 0.9855 | 0.9444 | 0.9645 |
| MLP + L1                  | 0.9649 | 0.9857 | 0.9583 | 0.9718 |
| MLP + L2                  | 0.9561 | 0.9855 | 0.9444 | 0.9645 |
| MLP + L1-L2                | 0.9561 | 0.9855 | 0.9444 | 0.9645 |
| Final Combined Model (Dropout + L2 + Early Stopping) | 0.9649 | 0.9857 | 0.9583 | 0.9718 |

The full breakdown, training curves, and confusion matrices for every model are in the notebook.

## Important Insights

- The SLP baseline, with only 31 parameters, already achieves over 92% test accuracy, but it learns
  only a linear decision boundary.
- Adding hidden layers (MLP) with ReLU activation gave the highest test accuracy, precision, recall
  and F1-score among all models trained in this project.
- Tanh and Sigmoid hidden-layer activations were also tested; the actual comparison and the winning
  activation are shown in the notebook (Task 3).
- Early Stopping, Dropout, and L1/L2/L1-L2 regularization were each tested as ways to control
  overfitting. Their actual effect on the train-validation gap and test performance is compared in
  Tasks 4, 5 and 6.
- For a clinical decision-support tool, recall matters a great deal because a false negative means a
  malignant case is predicted as benign, which can be more costly than a false positive. This is
  discussed in detail in the "Clinical Insight for Medical Diagnosis Support" section of the
  notebook.

## Project Files

```
DL_PR1/
│
├── Breast_cancer.ipynb        Full Jupyter notebook (all 7 tasks, executed with real results)
├── Breast_cancer.html          HTML export of the executed notebook
├── README.md            This file
├── requirements.txt     Python package requirements
├── plots/                Saved plots (PNG) from every task
└── video link            Link to the project walkthrough video
```

## How to Run

1. Create and activate a virtual environment (optional but recommended).
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook Breast_cancer.ipynb
   ```
4. Run all cells from top to bottom (Restart & Run All). The notebook does not need any manual
   changes between cells.

## Video Explanation Link

https://drive.google.com/file/d/1zpwE6kXhGyoTzOHd5nvQvi7cJQbISIUQ/view?usp=sharing
