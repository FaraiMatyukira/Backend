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

loaded_model = joblib.load('model_filename.joblib')
new_data = pd.read_csv("Grobler Boerdery_ndvi_Area3.csv",sep = ";")
# Calculate MSAVI
new_data['MSAVI'] = (2 * new_data['sur_refl_b02'] + 1 - np.sqrt((2 * new_data['sur_refl_b02'] + 1)**2 - 8 * (new_data['sur_refl_b02'] - new_data['sur_refl_b01']))) / 2
# Extract the features for prediction
X_test_data = new_data[['EVI', 'NDVI', 'MSAVI', 'sur_refl_b01', 'sur_refl_b02']]

# Standardize the test data using the same scaler used for training
scaler = StandardScaler()
X_test_data_scaled = scaler.transform(X_test_data)
# Assuming 'new_data' is a numpy array or DataFrame with the same features as used during training
predictions = loaded_model.predict(X_test_data_scaled)
predicted_categories = np.argmax(predictions, axis=1)

# Convert predicted categories back to original NDVI labels
predicted_labels = np.array([reverse_categorize_msavi(category) for category in predicted_categories])

# Print the predicted labels and original NDVI values for each data point
for i, (label, msavi_value) in enumerate(zip(predicted_labels, new_data['MSAVI'])):
    print(f'Data Point {i+1}: Predicted MSAVI Category - {label}, Original MSAVI Value - {msavi_value}')


