# Data Preprocessing Guide

**Preprocessing** = the boring but necessary part before the "cool" ML starts.

Raw data is usually messy: missing values, mixed types (numbers + categories), weird scales. If you throw this directly into a model, it will either complain, give garbage results, or both.

So preprocessing is about making the data clean and consistent → so the model can actually learn patterns instead of noise.

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
- Normalization → squashes values between min and max. Nice if features are kind of normally distributed.
- Standardization → subtract mean, divide by variance. Works well most of the time, a safe bet.

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

```python
data = pd.read_csv("file_name.csv")
```

Assumes your dataset is a CSV file. `data` is now a pandas DataFrame.

### 3. Split features (X) and target (y)

```python
X = data.iloc[:, :-1].values   # all columns except last → features
y = data.iloc[:, -1].values    # last column → target
```

- `.iloc[:, :-1]` → select all rows, all columns except last
- `.iloc[:, -1]` → select all rows, only last column
- `.values` → converts DataFrame to NumPy array (many ML models prefer arrays)

### 4. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- Splits data into training (80%) and test (20%)
- `random_state=42` → ensures reproducibility
- Why split early? → to avoid data leakage. All preprocessing must "learn" only from training data

### 5. Check Missing Values

```python
print(data.isnull().sum())
```

Shows how many missing values per column. Missing values must be handled before feeding data into a model.

### 6. Imputation (filling missing values)

```python
imputer = SimpleImputer(strategy="mean")

# fit only on training data
X_train[:, 1:3] = imputer.fit_transform(X_train[:, 1:3])

# transform test data with same stats (no cheating!)
X_test[:, 1:3] = imputer.transform(X_test[:, 1:3])
```

- Replaces missing values with the mean of each column
- Other strategies: `"median"`, `"most_frequent"`
- Fit on training set only, then apply to test → avoids leaking information from test set

### 7. Encoding (categorical → numbers)

```python
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), ["age"])],
    remainder="passthrough"
)

X_train = np.array(ct.fit_transform(X_train))
X_test = np.array(ct.transform(X_test))
```

- OneHotEncoder turns categories into 0/1 columns
- Example: `"red"`, `"blue"`, `"green"` → `[1,0,0]`, `[0,1,0]`, `[0,0,1]`
- `fit_transform` on training data, then `transform` on test data
- `remainder="passthrough"` → keeps other columns as they are

> **Note:** Here we used "age" just as an example of a categorical column; in real data you'd encode actual categorical features like Sex, Embarked, etc.

### 8. Scaling Numeric Features

```python
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

- Standardization = subtract mean, divide by variance
- Makes features comparable (so one column with huge numbers doesn't dominate)
- Again: fit only on train, transform test

## In Short

Preprocessing isn't about making the data look nice in a table, it's about survival.

- **Without it:** models might explode, refuse to converge, or just give nonsense
- **With it:** training is smoother, results are more reliable

Think of it this way → preprocessing is like washing vegetables before cooking.  
You can skip it, but you'll regret it later.