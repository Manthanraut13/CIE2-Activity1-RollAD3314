# CIE 2 – Activity 1: Multiple Linear Regression Worksheet

> **Predicting car fuel efficiency (mpg) from engine and vehicle characteristics**

| | |
|---|---|
| **Institution** | Zeal College of Engineering and Research, Narhe, Pune – 41 |
| **Department / Class** | AI & DS, T.Y. B.Tech, Semester V (Academic Year 2026–27) |
| **Course** | Regression and Classification Algorithms |
| **Activity** | CIE 2, Activity 1 (Unit I – Regression Foundations & Linear Regression), 5 marks |
| **Student** | Manthan Raut |
| **Roll No. / Division** | AD3314 / C |
| **Submission GitHub ID** | [https://github.com/complexiota](https://github.com/complexiota) |

---

## 1. Objective

Take the role of a data analyst at an automobile consultancy and explain how several design factors *together* drive fuel efficiency. The work covers:

- Solving the least-squares normal equations **by hand** for a small sample with two features, and verifying the result with a library.
- Comparing a **single-feature** model with a **multiple-feature** model.
- Interpreting the coefficients and intercept, checking the assumptions of linear regression (including **multicollinearity**), and evaluating the model on unseen data.

## 2. Problem statement

> Which car characteristics explain fuel efficiency, and can we forecast mpg for a car design before it is built?

| Item | Definition |
|------|-----------|
| **User / decision** | Product planners estimating the mpg impact of design choices |
| **Target (y)** | `mpg` – miles per gallon (continuous) |
| **Features (X)** | `displacement` (cu. in.), `horsepower` (hp), `weight` (lbs), `acceleration` (sec, 0–60 mph), `model_year` (70–82) |
| **Success metric** | Test RMSE (in mpg) and test R², which must beat a "predict the mean" baseline |

## 3. Dataset

- **Name:** Auto MPG
- **Original source:** Quinlan, R. (1993). *Auto MPG*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/9/auto+mpg
- **CSV mirror used in the notebook:** https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv
- **File type / size:** CSV, 398 rows × 9 columns (**392 rows after cleaning**)
- **Columns not used as features:** `name` (identifier/text), `origin` (categorical), `cylinders` (discrete and nearly duplicates `displacement`)
- **Data ethics:** Public, non-personal data; no restricted or confidential information.

### Data dictionary

| Column | Meaning | Unit | Range in data | Role |
|--------|---------|------|---------------|------|
| `mpg` | Fuel efficiency | miles per gallon | 9.0 – 46.6 | **Target** |
| `displacement` | Engine volume | cubic inches | 68 – 455 | Feature |
| `horsepower` | Engine power | hp | 46 – 230 | Feature |
| `weight` | Vehicle weight | lbs | 1613 – 5140 | Feature |
| `acceleration` | Time from 0 to 60 mph | seconds | 8.0 – 24.8 | Feature |
| `model_year` | Year of manufacture | 70–82 (19xx) | 70 – 82 | Feature |

**Data quality checks:** no duplicate rows; 6 missing values, all in `horsepower`.

## 4. Repository structure

```
CIE2-Activity1-MLR-<RollNo>/
├── README.md                    # this file
├── RCA_Activity_one.ipynb       # executed notebook (all outputs visible)
├── requirements.txt             # library versions used
├── worksheet/                   # manual calculation (typed PDF or photos)
└── figures/                     # saved plots (optional)
```

## 5. Work completed

| Step | What was done |
|------|---------------|
| 1. Problem framing | Defined user, X, y and success metric |
| 2. Data inspection and cleaning | Data types, summary statistics, missing values, duplicates; **6 rows (1.5%)** with missing `horsepower` removed |
| 3. Exploratory analysis | Scatter plots of mpg against each feature; labelled correlation heatmap; strongly correlated feature pairs identified |
| 4. Manual worksheet | 6 observations, features `weight` (in 1000 lbs) and `horsepower`; design matrix, XᵀX, Xᵀy, Gauss-Jordan elimination, b0/b1/b2, one prediction and residual |
| 5. Library models | 80:20 split (seed 42); mean baseline; single-feature model; multiple linear regression; scaler fit on training data only; hand calculation verified with scikit-learn |
| 6. Interpretation | Fitted equation, coefficients in dataset units, intercept meaning, standardised coefficients |
| 7. Predictions | Actual vs predicted on the test set |
| 8. Evaluation | MAE, MSE, RMSE, R², adjusted R²; single vs multiple comparison table |
| 9. Diagnostics | Residual plot, residual histogram, Q-Q plot, predicted-vs-actual plot, Shapiro-Wilk, Breusch-Pagan, VIF, drop-one-feature experiment |
| 10. New prediction | One new car inside the training range, with a range check |

**Environment:** Python 3.13, numpy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, statsmodels 0.15.0, scipy 1.16.3. Random seed: `42`.

## 6. Exploratory findings

| Feature | Correlation with mpg | Reading |
|---------|----------------------|---------|
| `weight` | −0.832 | Strong negative: heavier cars get fewer mpg |
| `displacement` | −0.805 | Strong negative |
| `horsepower` | −0.778 | Strong negative |
| `model_year` | +0.581 | Moderate positive: newer cars are more efficient |
| `acceleration` | +0.423 | Weak-to-moderate positive |

**Strongly correlated feature pairs (|r| > 0.8), a sign of multicollinearity:**

| Pair | r |
|------|---|
| `displacement` and `weight` | 0.93 |
| `displacement` and `horsepower` | 0.90 |
| `horsepower` and `weight` | 0.86 |

These three features largely carry the same information (engine and vehicle size). A visible relationship or high correlation does **not** prove causation; for example, weight may stand in for vehicle class.

## 7. Manual calculation summary

Sample: 6 randomly selected observations (`random_state=42`), with `weight` expressed in thousands of lbs. Full working is in [`worksheet/`](./worksheet) and in the notebook.

| obs | x1 = weight (1000 lbs) | x2 = horsepower | y = mpg |
|-----|------------------------|-----------------|---------|
| 1 | 2.189 | 69 | 26.0 |
| 2 | 2.795 | 115 | 21.6 |
| 3 | 1.800 | 60 | 36.1 |
| 4 | 1.955 | 70 | 26.0 |
| 5 | 2.790 | 86 | 27.0 |
| 6 | 2.155 | 75 | 28.0 |

**XᵀX** =

```
[[    6.0000     13.6840     475.0000]
 [   13.6840     32.0939    1118.8810]
 [  475.0000   1118.8810   39507.0000]]
```

**Xᵀy** = `[164.700, 368.766, 12686.000]`, and det(XᵀX) = 2518.0065 (non-zero, so a unique solution exists).

Solved by Gauss-Jordan elimination (matrix-inverse method gives the same answer):

| Coefficient | Value |
|-------------|-------|
| b0 (intercept) | 43.0952 |
| b1 (weight, per 1000 lbs) | −1.2099 |
| b2 (horsepower, per hp) | −0.1628 |

**Sample equation:** `mpg = 43.095 + (−1.210) × weight_1000lb + (−0.163) × horsepower`

**Prediction and residual (observation 1):** predicted 29.216 mpg, actual 26.000 mpg, residual (actual − predicted) = **−3.216 mpg**.

**Library check:** scikit-learn on the same 6 rows and 2 features gives `[43.095157, −1.209856, −0.162769]`, identical to the hand calculation (match = True).

## 8. Key results

**Split:** 313 training and 79 testing observations (80:20, `random_state=42`).

### Fitted equation (full training data, 5 features)

```
mpg = -15.3516 - 0.00073·displacement - 0.00197·horsepower
      - 0.00658·weight + 0.06245·acceleration + 0.76087·model_year
```

| Feature | Raw coefficient | Standardised coefficient (per 1 SD) |
|---------|-----------------|--------------------------------------|
| weight | −0.006582 | **−5.528** |
| model_year | +0.760870 | **+2.758** |
| acceleration | +0.062448 | +0.176 |
| horsepower | −0.001974 | −0.075 |
| displacement | −0.000727 | −0.075 |
| **Intercept** | −15.3516 | – |

**How to read the coefficients (holding the other features constant):**

- **weight:** each extra 1 lb lowers predicted mpg by about 0.0066, so an extra 1000 lbs lowers it by about 6.6 mpg.
- **model_year:** each later model year adds about 0.76 mpg.
- **acceleration:** a positive sign, because slower 0-60 times go with smaller, more economical cars.
- **horsepower and displacement:** very small effects *once weight is in the model*, because they overlap with weight (see VIF below). Their near-zero values do not mean engine size is irrelevant to mpg.
- **Intercept (−15.35):** the predicted mpg for a car with zero weight, zero horsepower and model year 0. This is physically meaningless (even negative), so it is only a mathematical anchor.
- **Relative influence:** use the standardised column. `weight` and `model_year` dominate. This shows association, not causation.

### Test-set comparison

| Model | MAE | MSE | RMSE | R² | Adj. R² |
|-------|-----|-----|------|----|---------|
| Mean baseline | 5.881 | 51.620 | 7.185 | −0.011 | −0.011 |
| Single feature (`weight`) | 3.464 | 17.693 | 4.206 | 0.653 | 0.649 |
| **Multiple (5 features)** | **2.513** | **10.579** | **3.252** | **0.793** | **0.779** |

Single-feature model: `mpg = 47.2005 − 0.007904 × weight` (intercept 47.2005, coefficient −0.007904).

**Interpretation:** The mean baseline explains nothing on unseen data (R² ≈ 0). Weight alone already explains about 65% of the variation in mpg. Adding the other four features raises that to about 79% and cuts the typical error (RMSE) from 4.2 to 3.3 mpg. Adjusted R² also rises (0.649 → 0.779), so the extra features add real information rather than just complexity. In plain terms, a typical prediction is off by roughly ±3 mpg.

### Actual vs predicted (sample of the test set)

| Actual mpg | Predicted mpg | Error |
|------------|---------------|-------|
| 15.0 | 12.82 | +2.18 |
| 26.0 | 25.94 | +0.06 |
| 34.1 | 34.01 | +0.09 |
| 19.4 | 23.60 | −4.20 |
| 27.4 | 27.87 | −0.47 |
| 26.0 | 28.51 | −2.51 |
| 15.0 | 12.44 | +2.56 |
| 22.0 | 23.76 | −1.76 |
| 27.0 | 29.38 | −2.38 |
| 16.9 | 16.43 | +0.47 |
| 29.0 | 30.41 | −1.41 |
| 31.6 | 29.84 | +1.76 |

Most predictions are within about 2–3 mpg; the largest miss in this sample is −4.20 mpg.

### Multicollinearity (VIF on training features)

| Feature | VIF | Reading |
|---------|-----|---------|
| displacement | 10.11 | Serious (> 10) |
| weight | 9.94 | High (> 5), near serious |
| horsepower | 9.34 | High (> 5), near serious |
| acceleration | 2.68 | Acceptable |
| model_year | 1.28 | No overlap |

**Interpretation:** `displacement`, `weight` and `horsepower` overlap heavily (consistent with r = 0.86–0.93), so their *individual* coefficients are unstable and hard to trust, even though the model predicts well as a whole.

**Drop-one experiment** (remove the highest-VIF feature, `displacement`):

| Model | MAE | MSE | RMSE | R² | Adj. R² |
|-------|-----|-----|------|----|---------|
| All 5 features | 2.513 | 10.579 | 3.252 | 0.793 | 0.779 |
| Without `displacement` | 2.512 | 10.553 | 3.249 | 0.793 | 0.782 |

Dropping `displacement` gives essentially the same accuracy (RMSE 3.249 vs 3.252) and a slightly higher adjusted R² (0.782 vs 0.779). So `displacement` is redundant and can be removed for a simpler, more stable model.

### Residual analysis and assumptions (test set)

| Check | Result |
|-------|--------|
| Shapiro-Wilk (normality of residuals) | p = 0.8734, no evidence against normality |
| Breusch-Pagan (constant variance) | p = 0.0784, not significant at 5%, but borderline |
| Residual-vs-predicted plot | Residuals are positive at low and high predicted mpg and mostly negative in the middle, a gentle curve |
| Predicted-vs-actual plot | Points follow the 45° line closely; very high-mpg cars (about 35+) are under-predicted |

| Assumption | Evidence | Verdict |
|-----------|----------|---------|
| Linearity | Curved residual pattern; scatter plots curve slightly | **Doubtful**: a straight line misses some curvature |
| Independence of errors | Each car is a separate record, no time-series order | Reasonable |
| Constant variance | Breusch-Pagan p = 0.0784 | Mostly reasonable, borderline |
| Approximately normal errors | Histogram roughly symmetric; Shapiro-Wilk p = 0.87 | Reasonable |
| No severe multicollinearity | VIF about 9–10 for three features | **Doubtful**: handled by dropping `displacement` |

### New prediction

Inputs are inside the training range (no extrapolation warning was raised).

| displacement | horsepower | weight | acceleration | model_year | Predicted mpg |
|--------------|-----------|--------|--------------|-----------|---------------|
| 150 | 100 | 2800 | 15 | 78 | **26.2 mpg** (typical error about ±3.3 mpg) |

## 9. Decision log (summary)

| # | Decision | Alternatives | Reason |
|---|----------|--------------|--------|
| 1 | Used Auto MPG | Concrete strength, California housing | Real multicollinearity; small enough for manual work |
| 2 | Dropped `name`, `origin`, `cylinders` | Keep `cylinders`, one-hot `origin` | Identifier; `cylinders` duplicates `displacement`; simpler model |
| 3 | Removed 6 rows with missing `horsepower` | Median imputation | Only 1.5% of rows; avoids artificial values |
| 4 | Kept outliers | Remove | They are real cars, not errors |
| 5 | 80:20 split, seed 42 | 70:30, cross-validation | Standard and reproducible |
| 6 | Scaler fit on training data only | Fit on all data | Prevents test leakage |
| 7 | Baselines: mean predictor and best single feature (`weight`) | None | A model must beat a simple benchmark |
| 8 | Tested removing `displacement` (VIF 10.11) | Keep all, combine features | RMSE 3.249 vs 3.252 and adj. R² 0.782 vs 0.779, so it adds nothing |
| 9 | Improvement tried: drop the redundant feature | `log(mpg)`, squared-weight term | Cheapest fix for multicollinearity; the curved residuals suggest a non-linear term is the next step to try |

## 10. Conclusion

The Auto MPG data (392 cars after cleaning) was modelled with multiple linear regression on five numeric features. On unseen data the model reaches **RMSE 3.25 mpg, R² 0.793 and adjusted R² 0.779**, clearly beating the mean baseline (RMSE 7.19, R² −0.011) and the weight-only model (RMSE 4.21, R² 0.653). Weight is the strongest driver (standardised coefficient −5.53), followed by model year (+2.76); displacement, horsepower and weight overlap heavily (VIF about 9–10), so their individual coefficients should not be over-interpreted, and `displacement` can be dropped without losing accuracy. Residuals are roughly normal but show mild curvature, so a linear model is only an approximation. For a new car (150 cu. in., 100 hp, 2800 lbs, 15 s, year 78) the model predicts about **26.2 mpg**, give or take about 3 mpg.

## 11. Limitations and responsible use

- A linear model can miss non-linear effects: the residual plot is curved and high-mpg cars are under-predicted.
- The data covers cars from 1970–1982 only. The model **must not** be used for modern cars, hybrids or electric vehicles, or for inputs outside the training range.
- Correlated features make individual coefficients unstable; coefficients show association, **not causation**.
- A prediction is an estimate, not a guarantee. Wrong predictions could mislead design or cost decisions but do not directly affect individuals.

## 12. How to run

1. Open `RCA_Activity_one.ipynb` in [Google Colab](https://colab.research.google.com) (or Jupyter).
2. Run **Runtime → Run all**. The dataset is downloaded automatically from the CSV link above.
3. For a local run: `pip install -r requirements.txt`, then open the notebook.

**Libraries:** pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels (exact versions in `requirements.txt`).

## 13. References

1. Quinlan, R. (1993). *Auto MPG* dataset. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/9/auto+mpg
2. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825–2830.
3. Seabold, S. and Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. *Proc. 9th Python in Science Conf.*

---

*Submitted for CIE 2 evaluation. Faculty submission GitHub ID: [complexiota](https://github.com/complexiota). All work was completed independently; decisions and explanations are in my own words.*
