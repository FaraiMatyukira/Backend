import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



# Convert predicted categories back to original NDVI labels
def reverse_categorize_msavi(category):
    if category == 0:
        return "Barren"
    elif category == 1:
        return "Germination stage"
    elif category == 2:
        return "Leaf development stage"
    elif category == 3:
        return "Vegetation covers soil"

    else:
        return "Undefined"
def reverse_categorize_ndvi(category):
    if category == 0:
        return "Barren"
    elif category == 1:
        return "Sparse Vegetation"
    elif category == 2:
        return "Dense Vegetation"
    else:
        return "Undefined"  
    
def reverse_categorize_evi(evi_value):
    if evi_value == 0:
        return "Low Vegetation"
    elif evi_value == 1:
        return "Moderate Vegetation"
    else:
        return "High Vegetation"
    
class farm_Classifier():
    def MSAVI(self):
        loaded_model = joblib.load(r'C:\Users\S_CSIS-PostGrad\Documents\Hounors\Research Projects\Artefact\Backend\classifiers\MSAVI.joblib')
        new_data = np.array([[758.4887218045112,3395.864661654135,359.0902255639097,1192.6691729323309, 3]])
        # Standardize the test case using the same scaler used for training
        scaler = StandardScaler()
        scaler.fit(new_data)
        test_case_scaled = scaler.transform(new_data)

        # Make predictions
        predicted_category = loaded_model.predict(test_case_scaled)
        predicted_category_index = np.argmax(predicted_category)
        print(predicted_category_index)
        # 4297.3007518796985
        # Convert the predicted category index to NDVI label
        predicted_label = reverse_categorize_msavi(predicted_category_index)
        return predicted_label
        # new_data = np.array([[643.06, 2550.2599999999998, 292.28999999999996, 1959.3499999999997, 3]])
        # # new_data = pd.read_csv("Grobler Boerdery_ndvi_Area3.csv",sep = ";")
        # # Calculate MSAVI
        # new_data['MSAVI'] = (2 * new_data['sur_refl_b02'] + 1 - np.sqrt((2 * new_data['sur_refl_b02'] + 1)**2 - 8 * (new_data['sur_refl_b02'] - new_data['sur_refl_b01']))) / 2
        # # Extract the features for prediction
        # X_test_data = new_data[['EVI', 'NDVI', 'MSAVI', 'sur_refl_b01', 'sur_refl_b02']]

        # # Standardize the test data using the same scaler used for training
        # scaler = StandardScaler()
        # # Fit the scaler on your training data
        # scaler.fit(X_test_data)
        # X_test_data_scaled = scaler.transform(X_test_data)
        # # Assuming 'new_data' is a numpy array or DataFrame with the same features as used during training
        # predictions = loaded_model.predict(X_test_data_scaled)
        # predicted_categories = np.argmax(predictions, axis=1)

        # # Convert predicted categories back to original NDVI labels
        # predicted_labels = np.array([reverse_categorize_msavi(category) for category in predicted_categories])

        # # Print the predicted labels and original NDVI values for each data point
        # for i, (label, msavi_value) in enumerate(zip(predicted_labels, new_data['MSAVI'])):
        #     print(f'Data Point {i+1}: Predicted MSAVI Category - {label}, Original MSAVI Value - {msavi_value}')



    def NDVI(self):
        loaded_model = joblib.load(r'C:\Users\S_CSIS-PostGrad\Documents\Hounors\Research Projects\Artefact\Backend\classifiers\NDVI.joblib')
        data = pd.read_csv(r"C:\Users\S_CSIS-PostGrad\Documents\Hounors\Research Projects\Artefact\Backend\predictors\data\Grobler Boerdery_ndvi_Area3.csv",sep = ";")
        data['EVI']= data['EVI']/10000
        new_data = np.array([[758.4887218045112,3395.864661654135,359.0902255639097,1192.6691729323309, 3]])
        # Standardize the test case using the same scaler used for training
        scaler = StandardScaler()
        scaler.fit(new_data)
        test_case_scaled = scaler.transform(new_data)

        # Make predictions
        predicted_category = loaded_model.predict(test_case_scaled)
        predicted_category_index = np.argmax(predicted_category)
        print(predicted_category_index)
        # 4297.3007518796985
        # Convert the predicted category index to NDVI label
        predicted_label = reverse_categorize_ndvi(predicted_category_index)
        return predicted_label
        # new_data = np.array([[643.06, 2550.2599999999998, 292.28999999999996, 1959.3499999999997, 3]])
        # # new_data = pd.read_csv("Grobler Boerdery_ndvi_Area3.csv",sep = ";")
        # # Extract the features for prediction
        # X_test_data = new_data[['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07', 'Month']]
        # # Fit the scaler on your training data
        #   # Standardize the test data using the same scaler used for training
        # scaler = StandardScaler()
        # scaler.fit(X_test_data)
        # # Standardize the test data using the same scaler used for training
        # X_test_data_scaled = scaler.transform(X_test_data)

        # # Make predictions
        # predictions = loaded_model.predict(X_test_data_scaled)
        # predicted_categories = np.argmax(predictions, axis=1)

        # # Convert predicted categories back to original NDVI labels
        # predicted_labels = np.array([reverse_categorize_ndvi(category) for category in predicted_categories])

        # # Print the predicted labels and original NDVI values for each data point
        # for i, (label, ndvi_value) in enumerate(zip(predicted_labels, new_data['NDVI'])):
        #     print(f'Data Point {i+1}: Predicted NDVI Category - {label}, Original NDVI Value - {ndvi_value}')



    def EVI(self):
        loaded_model = joblib.load(r'C:\Users\S_CSIS-PostGrad\Documents\Hounors\Research Projects\Artefact\Backend\classifiers\EVI.joblib')
        # Extract the features for prediction
        new_data = np.array([[758.4887218045112,3395.864661654135,359.0902255639097,1192.6691729323309, 3]])
        # Standardize the test case using the same scaler used for training
        scaler = StandardScaler()
        scaler.fit(new_data)
        test_case_scaled = scaler.transform(new_data)

        # Make predictions
        predicted_category = loaded_model.predict(test_case_scaled)
        predicted_category_index = np.argmax(predicted_category)
        print(predicted_category_index)
    	# 4297.3007518796985
        # Convert the predicted category index to NDVI label
        predicted_label = reverse_categorize_evi(predicted_category_index)  
        return predicted_label
        # # X_test_data = new_data[['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07', 'Month']]
        # # Fit the scaler on your training data
        # # Standardize the test data using the same scaler used for training
        # scaler = StandardScaler()
        # scaler.fit(new_data)
        # # Standardize the test data using the same scaler used for training
        # X_test_data_scaled = scaler.transform(new_data)

        # # Make predictions
        # predictions = loaded_model.predict(X_test_data_scaled)
        # predicted_categories = np.argmax(predictions, axis=1)

        # # Convert predicted categories back to original NDVI labels
        # predicted_labels = np.array([reverse_categorize_evi(category) for category in predicted_categories])

        # # Print the predicted labels and original NDVI values for each data point
        # for i, (label, ndvi_value) in enumerate(zip(predicted_labels, new_data['EVI'])):
        #     print(f'Data Point {i+1}: Predicted EVI Category - {label}, Original EVI Value - {ndvi_value}')

