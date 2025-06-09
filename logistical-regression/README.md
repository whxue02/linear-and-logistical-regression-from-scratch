# 🔎 Logistic Regression From Scratch — Breast Cancer Classifier

This project implements **logistic regression from scratch** using only NumPy — no `scikit-learn`! It trains and evaluates the model on the **Breast Cancer Wisconsin dataset** to classify tumors as **malignant** or **benign**.

---

## 📌 Features

- Logistic Regression implemented from scratch in Python
- Uses sigmoid activation and gradient descent
- Custom training loop with manual weight/bias updates
- Evaluates model accuracy on real-world biomedical data
- Achieves ~89% test accuracy!

---

## 🧠 Dataset

The model uses the [Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), available in `scikit-learn`.

- **Samples**: 569
- **Features**: 30 (e.g., radius, texture, symmetry)
- **Classes**:  
  - `0`: malignant  
  - `1`: benign

---

## 📂 Project Structure
├── lr.py # Custom Logistic Regression class
├── train.py # Loads data, trains the model, prints accuracy
├── README.md


---

## 📦 Dependencies

- Python 3.x
- NumPy
- scikit-learn
- matplotlib (optional for plotting)

Install with:

```bash
pip install numpy scikit-learn matplotlib
```

---

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/whxue02/linear-and-logistical-regression-from-scratch.git
cd logistic-regression-scratch
```

2. Run the main script:
```bash
python train.py
```
You should see an output like this:
```
0.8947368421052632
```

---

## 📘 How It Works
1. Initialize Parameters: Weights and bias set to zero
2. orward Pass: Compute predictions using sigmoid(X·w + b)
3. Loss: Not printed, but internally minimized via gradient descent
4. Backpropagation: Compute gradients and update weights
5. Prediction: Outputs 0 or 1 using a threshold of 0.5
