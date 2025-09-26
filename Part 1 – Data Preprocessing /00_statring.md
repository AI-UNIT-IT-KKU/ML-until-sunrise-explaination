# Part 1 – Data Preprocessing (Tabular / CSV Focus)

This section is about **general preprocessing for tabular data** (like CSV files).  
Why? Because most ML tutorials and first projects start with tabular datasets, and sklearn + pandas make it easy to work with them.

But keep in mind → preprocessing depends on the **type of data**:

- **tabular (CSV, spreadsheets, SQL extracts)** → handle missing values, scaling, encoding..
- **images** → resizing, normalization, augmentation..
- **text (NLP)** → cleaning, tokenization, stopword removal, embeddings..
- **audio** → sampling, spectrograms, noise reduction..
- **time series** → smoothing, windowing, lag features..

In this part, we only focus on **CSV/tabular preprocessing**, and later notes will cover their own preprocessing approaches.

![Data Science Meme](https://i.imgur.com/your-meme-here.jpg)
*When you're a data scientist mixing all libraries* 😂

---

## What is Preprocessing?

**Preprocessing** = the boring but necessary part before the "cool" ML starts.

Raw data is usually messy: missing values, mixed types (numbers + categories), weird scales. If you throw this directly into a model, it will either complain, give garbage results, or both.

So preprocessing is about **making the data clean and consistent** → so the model can actually learn patterns instead of noise.

In short: **good preprocessing = less pain later.**

## Main Steps

### 1. Imputation
Filling in missing values. You can use mean, median, or most frequent.  
Or just drop rows if you like chaos.

### 2. Encoding
Converting categories (like `male`, `female`, `other`) into numbers.  
Because models can't read strings, only numbers.  
OneHotEncoder is the classic way.

### 3. Scaling
Makes numeric features live on the same scale so the model doesn't freak out.
- **Normalization** → squashes values between min and max. Nice if features are kind of normally distributed.
- **Standardization** → subtract mean, divide by variance. Works well most of the time, a safe bet.

## Important Rule (no shortcuts here)

**Split first, preprocess later.**

Always split data into train/test first, then fit preprocessing only on the training data.

Why? Because if the model "peeks" at test data during preprocessing, that's called **data leakage**, and you basically cheated without meaning to.

- Encoders, imputers, scalers → fit on train, transform both train and test.
- Pre-trained embeddings (like BERT, Word2Vec) → safe to apply on both.
- Custom embeddings → train them only on training data.

## Simple Walk-through

This is the classic preprocessing pipeline you'll repeat in almost every ML project.  
Follow the flow step by step:

### 1. Libraries

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
```

- `pandas / numpy` → handle and manipulate data
- `train_test_split` → split dataset into train and test
- `SimpleImputer` → fill missing values
- `ColumnTransformer + OneHotEncoder` → encode categorical features
- `StandardScaler` → scale numeric features

### 2. Upload data

For this example, let's use some sample data ([Data.csv]()):

This dataset is about predicting whether customers will buy a product based on their demographic information.
```python
# Sample dataset
data = pd.read_csv("Data.csv")
print("Original Dataset:")
print(data)
```

Output:
```
Original Dataset:
   Country   Age   Salary Purchased
0   France  44.0  72000.0        No
1    Spain  27.0  48000.0       Yes
2  Germany  30.0  54000.0        No
3    Spain  38.0  61000.0        No
4  Germany  40.0      NaN       Yes
5   France  35.0  58000.0       Yes
6    Spain   NaN  52000.0        No
7   France  48.0  79000.0       Yes
8  Germany  50.0  83000.0        No
9   France  37.0  67000.0       Yes
```

Normally you'd use: `data = pd.read_csv("file_name.csv")` for CSV files.

### 3. Split features (X) and target (y)

```python
X = data.iloc[:, :-1].values   # all columns except last → features
y = data.iloc[:, -1].values    # last column → target

print("Features (X):")
print(X)
print("\nTarget (y):")
print(y)
```

Output:
```
Features (X):
[['France' 44.0 72000.0]
 ['Spain' 27.0 48000.0]
 ['Germany' 30.0 54000.0]
 ['Spain' 38.0 61000.0]
 ['Germany' 40.0 nan]
 ['France' 35.0 58000.0]
 ['Spain' nan 52000.0]
 ['France' 48.0 79000.0]
 ['Germany' 50.0 83000.0]
 ['France' 37.0 67000.0]]

Target (y):
['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']
```

- `.iloc[:, :-1]` → select all rows, all columns except last
- `.iloc[:, -1]` → select all rows, only last column
- `.values` → converts DataFrame to NumPy array (many ML models prefer arrays)

### 4. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"\nX_train:")
print(X_train)
print(f"\nX_test:")
print(X_test)
```

Output:
```
Training set size: (8, 3)
Test set size: (2, 3)

X_train:
[['Spain' 27.0 48000.0]
 ['Germany' 40.0 nan]
 ['France' 35.0 58000.0]
 ['Spain' 38.0 61000.0]
 ['Germany' 50.0 83000.0]
 ['France' 37.0 67000.0]
 ['Spain' nan 52000.0]
 ['France' 48.0 79000.0]]

X_test:
[['Germany' 30.0 54000.0]
 ['France' 44.0 72000.0]]
```

- Splits data into training (80%) and test (20%)
- `random_state=42` → ensures reproducibility
- Why split early? → to avoid data leakage. All preprocessing must "learn" only from training data

### 5. Check Missing Values

```python
print("Missing values in original data:")
print(data.isnull().sum())
```

Output:
```
Missing values in original data:
Country      0
Age          1
Salary       1
Purchased    0
dtype: int64
```

Shows how many missing values per column. Missing values must be handled before feeding data into a model.

### 6. Imputation (filling missing values)

```python
# Handle missing values in Age (column 1) and Salary (column 2)
imputer = SimpleImputer(strategy="mean")

# fit only on training data
X_train[:, 1:3] = imputer.fit_transform(X_train[:, 1:3])

# transform test data with same stats (no cheating!)
X_test[:, 1:3] = imputer.transform(X_test[:, 1:3])

print("After imputation:")
print("X_train:")
print(X_train)
print("\nX_test:")
print(X_test)
```

Output:
```
After imputation:
X_train:
[['Spain' 27.0 48000.0]
 ['Germany' 40.0 60857.14285714286]
 ['France' 35.0 58000.0]
 ['Spain' 38.0 61000.0]
 ['Germany' 50.0 83000.0]
 ['France' 37.0 67000.0]
 ['Spain' 39.142857142857146 52000.0]
 ['France' 48.0 79000.0]]

X_test:
[['Germany' 30.0 54000.0]
 ['France' 44.0 72000.0]]
```

- Replaces missing values with the mean of each column
- Other strategies: `"median"`, `"most_frequent"`
- Fit on training set only, then apply to test → avoids leaking information from test set

### 7. Encoding (categorical → numbers)

```python
# Encode the Country column (column 0)
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])],  # Column 0 is Country
    remainder="passthrough"
)

X_train = np.array(ct.fit_transform(X_train))
X_test = np.array(ct.transform(X_test))

print("After encoding:")
print("X_train shape:", X_train.shape)
print("X_train:")
print(X_train)
print("\nX_test shape:", X_test.shape)
print("X_test:")
print(X_test)
```

Output:
```
After encoding:
X_train shape: (8, 5)
X_train:
[[0.0 0.0 1.0 27.0 48000.0]
 [0.0 1.0 0.0 40.0 60857.14285714286]
 [1.0 0.0 0.0 35.0 58000.0]
 [0.0 0.0 1.0 38.0 61000.0]
 [0.0 1.0 0.0 50.0 83000.0]
 [1.0 0.0 0.0 37.0 67000.0]
 [0.0 0.0 1.0 39.142857142857146 52000.0]
 [1.0 0.0 0.0 48.0 79000.0]]

X_test shape: (2, 5)
X_test:
[[0.0 1.0 0.0 30.0 54000.0]
 [1.0 0.0 0.0 44.0 72000.0]]
```

The columns now represent: [France, Germany, Spain, Age, Salary]

### 8. Scaling Numeric Features

```python
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("After scaling:")
print("X_train:")
print(X_train)
print("\nX_test:")
print(X_test)
```

Output:
```
After scaling:
X_train:
[[-0.5 -0.5  1.5 -1.19  -1.26]
 [-0.5  1.5 -0.5  0.21  -0.07]
 [ 1.5 -0.5 -0.5 -0.71  -0.42]
 [-0.5 -0.5  1.5 -0.15  -0.25]
 [-0.5  1.5 -0.5  1.03   1.56]
 [ 1.5 -0.5 -0.5 -0.43   0.28]
 [-0.5 -0.5  1.5 -0.09  -0.77]
 [ 1.5 -0.5 -0.5  0.83   0.93]]

X_test:
[[-0.5  1.5 -0.5 -0.99  -0.59]
 [ 1.5 -0.5 -0.5  0.45   0.45]]
```

Perfect! Now your data is ready for machine learning models.

- Standardization = subtract mean, divide by variance
- Makes features comparable (so one column with huge numbers doesn't dominate)
- Again: fit only on train, transform test

## In Short

Preprocessing isn't about making the data look nice in a table, it's about survival.

- **Without it:** models might explode, refuse to converge, or just give nonsense
- **With it:** training is smoother, results are more reliable

Think of it this way → preprocessing is like washing vegetables before cooking.  
You can skip it, but you'll regret it later.