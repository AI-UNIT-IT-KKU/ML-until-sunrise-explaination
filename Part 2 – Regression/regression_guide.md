# Regression

A clean, professional walkthrough for your repo. Short sections, concrete explanations, runnable scripts, and the exact training files you provided included verbatim. Use it as a landing page for beginners who want clarity and practitioners who want correctness.

![Decision Tree: piecewise predictions](images/download%20(24).png)
![Random Forest: smoother fit](images/download%20(25).png)

## What regression is (and why you use it)

Regression predicts a continuous target such as price, temperature, or demand. You learn a function ŷ = f(x) and compare it to the truth y. The miss is the residual, ε = y - ŷ. If residuals are random around zero and don't show a pattern, you're modeling the signal; if they curve, trend, or fan out, your current model is missing structure.

## Simple linear regression (the straight-line baseline)

**Shape only:** y = b + ax

- Intercept b is the baseline when x = 0
- Slope a is "change in y per one unit of x"

**Plain-language assumptions:** the trend is approximately straight; errors hover around zero without a pattern; the spread of errors is similar across the range of x; observations are independent; extreme outliers can bend the line.

## Multiple linear regression (many features, same idea)

**Shape only:** y = b + a₁x₁ + a₂x₂ + ⋯ + aₙxₙ

Each aᵢ is the effect of xᵢ holding the others fixed.

Numeric features can go in directly (after basic cleaning). Categorical features must be turned into dummy variables with one-hot encoding so the model can learn a separate baseline per category. Drop one dummy per categorical feature to avoid the dummy variable trap (perfect multicollinearity). Scaling is not required for MLR to fit, but it can improve numerical stability, help regularized variants, and make coefficients more comparable when features live on very different scales. The quick test for "is a linear model appropriate?" is the residual plot; if it curves, consider a polynomial or non-linear model.

## Building the feature set (all-in vs stepwise)

"All-in" throws every reasonable feature into the design matrix and evaluates. "Backward elimination" removes the weakest features iteratively. "Forward selection" starts empty and adds the strongest. "Bidirectional" mixes both add/remove at each step. "Stepwise" generally refers to these add/remove procedures. Whatever you choose, use cross-validation as the referee rather than trusting a single split.

## Polynomial regression (linear model, non-linear shape)

**Shape only (one feature):**

y = b + a₁x + a₂x² + ⋯ + aₐxᵈ

You still train a linear model, just on expanded features (x, x², …, xᵈ). Low degrees capture gentle curvature; higher degrees add flexibility and overfitting risk. Scaling often helps because powers like x³ can explode numerically.

## Support Vector Regression (SVR) in one paragraph

SVR fits a function that keeps most points within an ε-insensitive tube; errors inside the tube don't count, and those outside are penalized by C. Kernels define the shape: linear for straight lines, polynomial for global curves and interactions, and RBF for smooth, local flexibility (intuitively, each training point contributes a bell-shaped influence; closer points matter more). Scaling is mandatory for SVR: scale both X and y. Train in scaled space, then inverse_transform predictions back to original units before plotting or reporting.

## Trees and forests (thresholds, interactions, and stability)

**Decision Trees (CART)** split features into regions and predict the average in each region. They don't need scaling. They're expressive and can overfit if grown deep. Set random_state for reproducibility since equal-quality splits can tie.

**Random Forests** average many decision trees trained on bootstrapped samples with random feature subsets. Averaging lowers variance and stabilizes performance. The big knobs are n_estimators and random_state. Like trees, forests don't need scaling.

![Decision Tree: piecewise predictions](images/download%20(24).png)
![Random Forest: smoother fit](images/download%20(25).png)

## Evaluation metrics you actually use

**RSS** (Residual Sum of Squares) sums squared residuals; **TSS** (Total Sum of Squares) measures total variation around the mean;

**R² = 1 - RSS/TSS** is the fraction of variance explained; **Adjusted R²** corrects the tendency of R² to inflate as you add features by penalizing complexity relative to sample size. In scikit-learn you'll commonly call `sklearn.metrics.r2_score(y_test, y_pred)`.

## Overfitting vs underfitting (and what to do)

- **Underfitting:** model is too simple; both train and test performance are poor
- **Overfitting:** model memorizes quirks; train looks great, test drops

**Practical fixes:** improve features; reduce flexibility (lower polynomial degree, shallower tree); add regularization (Ridge/Lasso for linear models); and pick settings with cross-validation. If diagnostics show a clear shape mismatch (e.g., curved residuals under a straight line), switch families rather than forcing a model to behave.

## Choosing between linear, polynomial, SVR, trees, and forests

Start with the simplest thing that could work. If the relationship is straight and interpretability matters, use linear or multiple linear. If residuals curve, try a low-degree polynomial. If you want a smooth, flexible baseline and you're comfortable tuning, use SVR with RBF. If you expect thresholds/interaction effects or want minimal preprocessing, probe with a decision tree; for accuracy and stability, jump to a random forest. When performance disappoints, tune first; if the residual patterns still scream "wrong shape," switch models.

## Categorical vs numeric features

Numeric features are used as-is. Categorical features require one-hot encoding. Drop one dummy per categorical feature to define the baseline and avoid perfect multicollinearity. A minimal pattern in scikit-learn looks like this:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.DataFrame({
    "size": [30, 45, 60, 80, 100],
    "city": ["A","A","B","B","C"],
    "rent": [1200, 1600, 1900, 2400, 2600]
})

X = df[["size","city"]]
y = df["rent"].values

ct = ColumnTransformer(
    [("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["city"])],
    remainder="passthrough"
)

X_enc = ct.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.4, random_state=0)

reg = LinearRegression().fit(X_train, y_train)
print("R^2:", reg.score(X_test, y_test))
```

## Your training scripts (verbatim)

These are your exact files for readers who want to run the same approach you used. Replace paths like `ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv` with your dataset.

### multiple_linear_regression.py

```python
# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```

### polynomial_regression.py

```python
# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Polynomial Regression model on the Training set
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)

# Predicting the Test set results
y_pred = regressor.predict(poly_reg.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```

### support_vector_regression.py

```python
# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Training the SVR model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```

### decision_tree_regression.py

```python
# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Decision Tree Regression model on the Training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```

### random_forest_regression.py

```python
# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```

## Quick answers to your notes

**When to pick SVR vs polynomial vs linear:** start linear; if residuals curve, try low-degree polynomial; if you need smooth flexibility and are okay with tuning, use SVR with RBF; if you expect thresholds or interactions, try trees/forests.

**Is feature scaling needed for Decision Trees or Random Forests?** No. They split on thresholds and don't depend on distances. SVR requires scaling for both X and y; remember to inverse-transform predictions.

**Why reshape in SVR but not in DT examples?** scikit-learn expects features as a 2-D array (n_samples, n_features). In some snippets you already had 2-D arrays; in SVR you started from a 1-D vector and reshaped to (-1,1).

**Model bad — tune or switch?** Tune sensible hyperparameters and improve features first; if residual patterns clearly show the model class is wrong (e.g., straight line for curved data), switch families.

**What is ensemble learning here?** Combining multiple trained models to make a final prediction; Random Forest averages many trees. Key params: n_estimators and random_state.

**What are RSS, TSS, R², and Adjusted R²?** RSS sums squared residuals; TSS is total variation around the mean; R² = 1 - RSS/TSS; Adjusted R² adds a penalty for extra predictors. In scikit-learn, use `r2_score(y_test, y_pred)`.
