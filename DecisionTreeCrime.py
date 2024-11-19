import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

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

# Check y values
min_instances = 3000
class_counts = y.value_counts()
mask = class_counts[y].values >= min_instances
X_filtered = X_scaled_imputed[mask]
y_filtered = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.25)
while any(value not in y_test.unique() for value in y_filtered.unique()):
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.25)

unique_classes_in_y = y_filtered.unique()
unique_classes_in_y_test = y_test.unique()

if set(unique_classes_in_y) == set(unique_classes_in_y_test):
    print("All classes are present in y_test.")
else:
    print("Warning: Not all classes are present in y_test.")

# Create a Decision Tree model
model = DecisionTreeClassifier(criterion="entropy")

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print("Accuracy on the testing set:", accuracy)
print("Precision on the testing set:", precision)
print("Recall on the testing set:", recall)
print("F1 score on the testing set:", f1)
