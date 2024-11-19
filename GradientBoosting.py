import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Load the dataset
data = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

seed = 10
subsample_fraction = .01

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

# Assuming 'category' is the target variable and other columns are features
X = filtered_data[['Occ_Year', 'Occ_Month', 'Occ_Day', 'Rptd_Year', 'Rptd_Occ_Month', 'Rptd_Occ_Day',
                   'TIME OCC', 'AREA', 'Rpt Dist No', 'Part 1-2', 'Premis Cd', 'LOCATION',
                    'LAT', 'LON']]
y = filtered_data['Crm Cd']

# Use SimpleImputer for imputation (fill missing values with mean for simplicity)
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()

# Standardize numerical features after imputation
X_scaled_imputed = scaler.fit_transform(X_imputed)

# Define the number of folds (k)
n_folds = 3  # You can adjust this based on your preference

# Initialize KFold
kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

# Create a Gradient Boosting Tree model
#gbt_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=seed)

# Initialize variables to track the number of instances for each class
min_instances = 100
class_counts = y.value_counts()

# Create a mask to filter out classes with less than min_instances instances
mask = class_counts[y].values >= min_instances

# Filter the data and labels
X_filtered = X_scaled_imputed[mask]
y_filtered = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X_scaled_imputed, y, test_size=0.25, random_state=seed)

# Create a Gradient Boosting Tree model
gbt_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=seed)

# Fit the model on the training data
gbt_model.fit(X_train, y_train)

# Predict on the test set
y_pred = gbt_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print("Accuracy on the testing set:", accuracy)
print("Precision on the testing set:", precision)
print("F1 score on the testing set:", f1)
