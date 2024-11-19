from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss, precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

# Load the dataset
data = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

# Specify the fraction of data to subsample (e.g., 0.1 for 10%)
seed = 10
subsample_fraction = 1

# Subsample the data
data = data.sample(frac=subsample_fraction, random_state=seed)


data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p')
data['DATE Rptd'] = pd.to_datetime(data['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p')

# Extract features from 'DATE OCC'
data['Occ_Year'] = data['DATE OCC'].dt.year
data['Occ_Month'] = data['DATE OCC'].dt.month
data['Occ_Day'] = data['DATE OCC'].dt.day

data['Rptd_Year'] = data['DATE Rptd'].dt.year
data['Rptd_Occ_Month'] = data['DATE Rptd'].dt.month
data['Rptd_Occ_Day'] = data['DATE Rptd'].dt.day

columns_to_check = ['Occ_Year', 'Occ_Month', 'Occ_Day','Rptd_Year','Rptd_Occ_Month','Rptd_Occ_Day',
                    'TIME OCC', 'AREA', 'Rpt Dist No','Part 1-2','Mocodes','Crm Cd','Premis Cd', 'LOCATION',
                    'Vict Age', 'Vict Sex', 'Vict Descent','Weapon Used Cd','Status','LAT', 'LON']

filtered_data = data.dropna(subset=columns_to_check).copy()

label_encoder = LabelEncoder()
filtered_data['LOCATION'] = label_encoder.fit_transform(filtered_data['LOCATION'])
filtered_data['Vict Sex'] = label_encoder.fit_transform(filtered_data['Vict Sex'])
filtered_data['Vict Descent'] = label_encoder.fit_transform(filtered_data['Vict Descent'])
filtered_data['Status'] = label_encoder.fit_transform(filtered_data['Status'])

# Assuming 'category' is the target variable and other columns are features
filtered_data['Mocodes_First4'] = filtered_data['Mocodes'].str[:4]
X = filtered_data[['Occ_Year', 'Occ_Month', 'Occ_Day','Rptd_Year','Rptd_Occ_Month','Rptd_Occ_Day', 
                   'TIME OCC', 'AREA', 'Rpt Dist No','Part 1-2','Mocodes_First4','Premis Cd','LOCATION', 
                   'Vict Age','Vict Sex','Vict Descent','Weapon Used Cd','Status','LAT', 'LON']]
y = filtered_data['Crm Cd']


# Use SimpleImputer for imputation (fill missing values with mean for simplicity)
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)
 
scaler = StandardScaler()

# Standardize numerical features after imputation
X_scaled_imputed = scaler.fit_transform(X_imputed)
# Define the number of folds (k)
n_folds = 3  # You can adjust this based on your preference

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

# Create a logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Initialize variables to track the number of instances for each class
min_instances = 3000

class_counts = y.value_counts()

# Create a mask to filter out classes with less than min_instances instances
mask = class_counts[y].values >= min_instances

# Filter the data and labels
X_filtered = X_scaled_imputed[mask]
y_filtered = y[mask]

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

# Iterate through the folds
for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X_filtered, y_filtered)):
    X_train, X_test = X_filtered[train_idx], X_filtered[test_idx]
    y_train, y_test = y_filtered.iloc[train_idx], y_filtered.iloc[test_idx]

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model for each fold
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    # Print the evaluation metrics for each fold
    print("Accuracy on the testing set:", accuracy)
    print("Precision on the testing set:", precision)
    print("Recall on the testing set:", recall)
    print("F1 score on the testing set:", f1)
    print("\n")

mean_accuracy = np.mean(accuracy_list)
mean_precision = np.mean(precision_list)
mean_f1 = np.mean(f1_list)
mean_recall = np.mean(recall_list)

# Print the mean metrics
print("Mean Accuracy across all folds:", mean_accuracy)
print("Mean Precision across all folds:", mean_precision)
print("Mean recall across all folds:", mean_recall)
print("Mean F1 score across all folds:", mean_f1)


# Extract feature names
feature_names = X.columns

# Get the coefficients from the trained logistic regression model
coefficients = model.coef_

# Create a DataFrame to store feature names and their corresponding coefficients
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients[0]})

# Sort the DataFrame by absolute coefficient values
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

# Plot the top N features
top_n = 25  # Choose the number of top features to plot
plt.figure(figsize=(12, 8))
plt.barh(coef_df['Feature'][:top_n], coef_df['Coefficient'][:top_n])
plt.xlabel('Coefficient Value')
plt.title('Top Features by Coefficient Value')
plt.show()

# Calculate the correlation matrix
correlation_matrix = X.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
plt.title('Correlation Heatmap of Features')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.show()