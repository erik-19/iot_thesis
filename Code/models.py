import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import MinMaxScaler, label_binarize
import joblib

# Load datasets
datatraining = pd.read_csv("/Users/erikvunsh/Desktop/thesis (1)/Data/datatraining.txt", sep=",")
datatesting = pd.read_csv("/Users/erikvunsh/Desktop/thesis (1)/Data/datatest.txt", sep=",")
datatesting2 = pd.read_csv("/Users/erikvunsh/Desktop/thesis (1)/Data/datatest2.txt", sep=",")
data = pd.concat([datatraining, datatesting, datatesting2])

data["Occupancy"] = data["Occupancy"].astype("category")
data["date"] = pd.to_datetime(data["date"], utc=True)

# Define helper functions
def second_day(x):
    return x.hour * 3600 + x.minute * 60 + x.second

def weekend_weekday(x):
    if x.weekday() == 5 or x.weekday() == 6:
        return "Weekend"
    else:
        return "Weekday"

def relevel_weekend(x):
    if x == "Weekend":
        return 0
    else:
        return 1

# Preprocess data
data_b = data.copy()
data_b["Time_stamp"] = data_b["date"].apply(second_day)
data_b["Week_status"] = data_b["date"].apply(weekend_weekday)
data_b["Week_status"] = data_b["Week_status"].apply(relevel_weekend)

data_b = pd.get_dummies(data_b, columns=['Week_status'], drop_first=True)

target_data = data_b['Occupancy']
data_b = data_b.drop(columns=['Occupancy', 'date'])

#Split data
data_train, data_temp, target_train, target_temp = train_test_split(data_b, target_data, test_size=0.3, random_state=42)
data_val, data_test, target_val, target_test = train_test_split(data_temp, target_temp, test_size=0.5, random_state=42)

#Gridseach with Cross-validation
param_grid = {
    'cart': {'max_depth': [3, 5, 7, None]},
    'rf': {'n_estimators': [50, 100, 200], 'max_features': ['sqrt', 'log2']}
}

lda = LinearDiscriminantAnalysis()
cart = DecisionTreeClassifier()
rf = RandomForestClassifier()

grid_search_result = {}

#Dont include LDA as does not have hyperparameters
lda.fit(data_train, target_train)
grid_search_result['lda'] = lda

for name, clf, params in [('cart', cart, param_grid['cart']), ('rf', rf, param_grid['rf'])]:
    grid_search = GridSearchCV(clf, params, cv=5, scoring='accuracy')
    grid_search.fit(data_train, target_train)
    grid_search_result[name] = grid_search.best_estimator_

lda = grid_search_result['lda']
cart = grid_search_result['cart']
rf = grid_search_result['rf']

lda_pred_test = lda.predict(data_test)
cart_pred_test = cart.predict(data_test)
rf_pred_test = rf.predict(data_test)

# Put a cap on the nightly CO2 readings (explained in thesis)
def filter_co2_readings(df):
    if not pd.api.types.is_datetime64_any_dtype(df['date_time']):
        df['date_time'] = pd.to_datetime(df['date_time'], utc=True)
    
    mask = df['date_time'].apply(lambda x: 21 <= x.hour or x.hour < 6 or (x.hour == 6 and x.minute < 30))
    
    df.loc[mask, 'CO2'] = 480
    
    return df



def preprocess_new_data(file_path):
    new_data = pd.read_csv(file_path, sep=',')
    #Drop irrelvant columns
    new_data = new_data.drop(columns=["OLD_TEMP", "P_sat", "vapor_pressure"])
    new_data["date_time"] = pd.to_datetime(new_data["date_time"], utc=True)

    new_data = filter_co2_readings(new_data)

    new_data_b = new_data.copy()
    new_data_b["Time_stamp"] = new_data_b["date_time"].apply(second_day)
    new_data_b["Week_status"] = new_data_b["date_time"].apply(weekend_weekday)
    new_data_b["Week_status"] = new_data_b["Week_status"].apply(relevel_weekend)
    new_data_b = pd.get_dummies(new_data_b, columns=['Week_status'], drop_first=True)
    
    week_status_columns = [col for col in new_data_b.columns if col.startswith('Week_status_')]
    for col in week_status_columns:
        new_data_b[col] = new_data_b[col].astype('category')

    new_data_b = new_data_b.drop(columns="date_time")
    
    #Scale values to the min and max ranges of the intial data
    min_max_scaler = MinMaxScaler()
    feature_ranges = {
        'Temperature': (data_b["Temperature"].min(), data_b["Temperature"].max()), 
        'Humidity': (data_b["Humidity"].min(), data_b["Humidity"].max()), 
        'Light': (data_b["Light"].min(), data_b["Light"].max()),
        'CO2': (data_b["CO2"].min(), data_b["CO2"].max()),
        'HumidityRatio': (data_b["HumidityRatio"].min(), data_b["HumidityRatio"].max()),
    }
    for feature in new_data_b.columns:
        if feature in feature_ranges:
            custom_range = feature_ranges.get(feature, (-1, 1))
            feature_data = new_data_b[[feature]].values.reshape(-1, 1)
            min_val, max_val = custom_range
            min_max_scaler.feature_range = (min_val, max_val)
            scaled_feature_data = min_max_scaler.fit_transform(feature_data)
            new_data_b[feature] = scaled_feature_data.flatten()
    

    return new_data, new_data_b

def transfer_learning(data_file, lda, cart, rf):
    data, data_b = preprocess_new_data(data_file)
    train, test = train_test_split(data_b, test_size=0.5, random_state=42)

    # Evaluate models on dataset before transfer learning
    lda_pred = lda.predict(test)
    cart_pred = cart.predict(test)
    rf_pred = rf.predict(test)

    print(f"\nData Evaluation before Transfer Learning for {data_file}:")
    print("LDA Predictions:", np.bincount(lda_pred))
    print("CART Predictions:", np.bincount(cart_pred))
    print("Random Forest Predictions:", np.bincount(rf_pred))

    # Transfer Learning
    pseudo_labels = rf.predict(train)
    train['Occupancy'] = pseudo_labels
    lda_tl = LinearDiscriminantAnalysis()
    lda_tl.fit(train.drop(columns='Occupancy'), train['Occupancy'])
    lda_tl_pred = lda_tl.predict(test)

    cart_tl = DecisionTreeClassifier()
    cart_tl.fit(train.drop(columns='Occupancy'), train['Occupancy'])
    cart_tl_pred = cart_tl.predict(test)

    rf_tl = RandomForestClassifier(n_estimators=100, max_features='sqrt')
    rf_tl.fit(train.drop(columns='Occupancy'), train['Occupancy'])
    rf_tl_pred = rf_tl.predict(test)

    print(f"\nData Evaluation after Transfer Learning for {data_file}:")
    print("LDA TL Predictions:", np.bincount(lda_tl_pred))
    print("CART TL Predictions:", np.bincount(cart_tl_pred))
    print("Random Forest TL Predictions:", np.bincount(rf_tl_pred))

    return lda_tl, cart_tl, rf_tl


data_files = [
    "/Users/erikvunsh/Desktop/thesis (1)/Data/thingy026_data.csv",
    "/Users/erikvunsh/Desktop/thesis (1)/Data/thingy008_data.csv",
    "/Users/erikvunsh/Desktop/thesis (1)/Data/thingy032_data.csv",
    "/Users/erikvunsh/Desktop/thesis (1)/Data/thingy034_data.csv",
    "/Users/erikvunsh/Desktop/thesis (1)/Data/thingy052_data.csv",
    "/Users/erikvunsh/Desktop/thesis (1)/Data/thingy063_data.csv"
]

for data_file in data_files:
    lda_tl, cart_tl, rf_tl = transfer_learning(data_file, lda, cart, rf)
    # Save the models
    joblib_file = f"{data_file.split('/')[-1].split('.')[0]}_rf_tl.pkl"
    joblib.dump(rf_tl, joblib_file)