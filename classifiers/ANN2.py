import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
def reverse_categorize_evi(evi_value):
    if evi_value == 0:
        return {
            "class": "Low Vegetation",
            "suggestion": [
                "Irrigation Assessment: Evaluate irrigation practices and consider adjusting water supply if necessary.",
                "Nutrient Management: Conduct soil tests to determine nutrient deficiencies and address them.",
                "Drought Resilience: Implement strategies to make the vegetation more resilient to drought conditions."
            ]
        }
    elif evi_value == 1:
        return {
            "class": "Low Vegetation",
            "suggestion": [
                "Irrigation Assessment: Evaluate irrigation practices and consider adjusting water supply if necessary.",
                "Nutrient Management: Conduct soil tests to determine nutrient deficiencies and address them.",
                "Drought Resilience: Implement strategies to make the vegetation more resilient to drought conditions."
            ]
        }
    else:
        return {
            "class": "High Vegetation",
            "suggestion": [
                "Monitor Growth: Keep an eye on vegetation growth and consider pruning or thinning if necessary.",
                "Nutrient Optimization: Ensure optimal nutrient levels for maximum plant health and productivity.",
                "Harvest Planning: Plan for the upcoming harvest and post-harvest activities."
            ]
        }
    
class EVI:
    def train_evi_model(self):
        # Load your CSV data into a Pandas DataFrame
        data = pd.read_csv(r"C:\Users\farai\OneDrive\Desktop\Documents\Hounors\Artifact\CropSenseAPI\Backend\classifiers\Potch farm data.csv", sep=";")

        # Define NDVI categories based on the provided ranges
        def categorize_evi(evi_value):
            if evi_value <= 0.2:
                return 0 #"Low Vegetation"
            elif 0.2 < evi_value <= 0.5:
                return 1 #"Moderate Vegetation"
            else:
                return 2 #"High Vegetation"
            
        data['EVI_Category'] = data['EVI'].apply(categorize_evi)
        # Split the data into features (X) and target (y)
        # Split the data into features (X) and target (y)
        X = data[['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07', 'Month']]
        y = data['EVI_Category']

        # Encode categorical labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Convert target to one-hot encoded labels
        y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=3)

        # Build a more complex ANN model with adjusted dropout rates
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dropout(0.4),  # Adjusted dropout rate
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),  # Adjusted dropout rate
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),  # Adjusted dropout rate
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model for more epochs
        model.fit(X_train_scaled, y_train_encoded, epochs=50, batch_size=32, validation_split=0.2)
        return model

        
    def predict_evi(self,bands,model):
        import numpy as np
        # Train the model (You can call this function once to train the model)
        # Load the pre-trained model
        # # Ensure the input is a 2D array
        
        bands = np.array([bands])

        # # Standardize the input data
        scaler = StandardScaler()
        scaler.fit(bands)
        bands_scaled = scaler.transform(bands)

        # Make predictions
        predictions = model.predict(bands_scaled)
        predicted_categories = np.argmax(predictions, axis=1)
        print(predicted_categories)
        # Convert the predicted category index to NDVI label
        predicted_label = reverse_categorize_evi(predicted_categories)

        print(f'Predicted NDVI Category: {predicted_label}')

        return predicted_label
    # import pandas as pd
 

    # # Load your new CSV data into a Pandas DataFrame
    # new_data = pd.read_csv("Grobler Boerdery_ndvi_Area3.csv",sep = ";")

    # # Extract the features for prediction
    # X_test_data = new_data[['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07', 'Month']]

    # # # Standardize the test data using the same scaler used for training
    # # scaler.fit(X_test_data)

    # X_test_data_scaled = scaler.transform(X_test_data)

    # # Make predictions
    # predictions = model.predict(X_test_data_scaled)
    # predicted_categories = np.argmax(predictions, axis=1)

    # # Convert predicted categories back to original NDVI labels
    # predicted_labels = np.array([reverse_categorize_ndvi(category) for category in predicted_categories])

    # # Print the predicted labels and original NDVI values for each data point
    # for i, (label, ndvi_value) in enumerate(zip(predicted_labels, new_data['NDVI'])):
    #     print(f'Data Point {i+1}: Predicted NDVI Category - {label}, Original NDVI Value - {ndvi_value}')




# Define a sample test case
# # test_case = np.array([[758.4887218045112,3395.864661654135,359.0902255639097,1192.6691729323309,2]])
# test_case = np.array([[500, 600, 700, 400, 6]])  # Example feature values for sur_refl_b01, sur_refl_b02, sur_refl_b03, sur_refl_b07, Month
#   # Example feature values for sur_refl_b01, sur_refl_b02, sur_refl_b03, sur_refl_b07, Month

# # Make predictions
# predicted_ndvi_category = predict_ndvi(test_case)

# print(f'Predicted NDVI Category: {predicted_ndvi_category}')