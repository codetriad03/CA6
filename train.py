import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Read data
df = pd.read_csv("data_processed.csv")

# Preparing target and features
y = df.pop("cons_general").to_numpy()
y[y < 4] = 0  # Label as 0 if less than 4
y[y >= 4] = 1  # Label as 1 if 4 or higher

X = df.to_numpy()
X = preprocessing.scale(X)  # Standardize features

# Impute missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X = imp.transform(X)

# Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Random Forest with 100 trees
yhat = cross_val_predict(clf, X, y, cv=5)

# Calculate metrics
acc = np.mean(yhat == y)
tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

# Save metrics to JSON file
with open("metrics.json", 'w') as outfile:
    json.dump({
        "accuracy": acc,
        "specificity": specificity,
        "sensitivity": sensitivity
    }, outfile)

# Visualize accuracy by region
score = yhat == y
score_int = [int(s) for s in score]
df['pred_accuracy'] = score_int

sns.set_color_codes("dark")
ax = sns.barplot(x="region", y="pred_accuracy", data=df, palette="Greens_d")
ax.set(xlabel="Region", ylabel="Model accuracy")
plt.savefig("by_region.png", dpi=80)
