# Linear Regression from Scratch - Olympic Team Medals Predictor 

This project implements **multivariate linear regression** using only NumPy to predict how many **Olympic medals** a country will win based on:
- Number of athletes sent
- Number of medals won in previous games

It includes **manual matrix math**, coefficient interpretation, and model evaluation using the **R² score**.

---

## 📊 Dataset

The dataset (`teams.csv`) contains information about Olympic participation over several years:

| team | year | athletes | events | age  | height | weight | prev_medals | medals |
|------|------|----------|--------|------|--------|--------|-------------|--------|
| AFG  | 1964 | 8        | 8      | 22.0 | 161.0  | 64.2   | 0.0         | 0      |
| AFG  | 2008 | 4        | 4      | 22.5 | 179.2  | 62.8   | 0.0         | 1      |
| ...  | ...  | ...      | ...    | ...  | ...    | ...    | ...         | ...    |

---

## 🧠 Project Workflow

### 1. Load & Prepare Data

```python
teams = pd.read_csv('teams.csv')
X = teams[["athletes", "prev_medals"]].copy()
y = teams[["medals"]].copy()
```
Add an intercept column manually to X:
```python
X["intercept"] = 1
X = X[["intercept", "athletes", "prev_medals"]]
```

### 2. Compute Regression Coefficients
Using the Normal Equation:
B = (XᵀX)⁻¹ Xᵀy
Where:
 - X is the design matrix (includes intercept, athletes, prev_medals)
 - y is the target vector (medals)
 - B is the coefficient vector we solve for

### 3. Make Predictions
```python
predictions = X @ B
```

### 4. Evaluate with R² Score
Using the R² Formula:
R² = 1 - SSR/SST
Where:
 - SSR = sum of squared residuals
 - SST = total sum of squares

Finding:
R² = 0.87  **Meaning: 87% of the variance in medal count isexplained by the model**

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/whxue02/linear-and-logistical-regression-from-scratch.git
cd linear-regression
```

2. Run the main script:
```bash
python lr.py
```





