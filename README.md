# Titanic-Survival-Prediction
Machine learning project predicting Titanic passenger survival using Random Forest classification.
Overview
Kaggle competition solution using feature engineering and optimized Random Forest to predict survival outcomes.
Features
Feature Engineering:

Title extraction from names (Mr., Mrs., Miss., Master)
FamilySize & IsAlone indicators
FarePerPerson & FareLog transformations
Sex_Pclass interaction features
Age bands and cabin deck extraction

Data Preprocessing:

Missing values filled with median/mode
One-hot encoding for categorical variables
Feature alignment between train/test sets

Model
pythonRandomForestClassifier(
    n_estimators=1200,
    max_depth=9,
    min_samples_split=4,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)
Custom threshold: 0.475
Usage
bashpip install pandas numpy scikit-learn matplotlib seaborn

# Download Kaggle dataset
kaggle competitions download -c titanic
Tech Stack
Python, pandas, numpy, scikit-learn, matplotlib, seaborn
Author
Huy Nguyen - UC Berkeley
