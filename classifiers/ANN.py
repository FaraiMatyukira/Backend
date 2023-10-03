import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
def reverse_categorize_ndvi(category):
    if category == 0:
        return  {
            "class": "Barren",
            "suggestion": [
                "Soil Testing: Conduct soil tests to assess nutrient deficiencies or pH imbalances.",
                "Soil Improvement: Add organic matter to improve soil quality."
                "Consider Alternative Crops: Evaluate whether other crop options or land use strategies might be more suitable for this area."
            ]
        },
    elif category == 1:
        return {
            "class": "Sparse Vegetation",
            "suggestion": [
                "Irrigation Management: Optimize irrigation practices to ensure adequate water supply.",
                "Nutrient Analysis: Conduct soil tests to determine nutrient needs and apply appropriate fertilizers.",
                "Pest and Weed Control: Implement pest and weed management strategies to promote healthier vegetation."
            ]
        }
    elif category == 2:
        return {
            "class": "Dense Vegetation",
            "suggestion": [
                "Pruning and Thinning: If applicable, consider thinning crowded areas to allow better light penetration.",
                "Disease Management: Monitor for plant diseases and take preventive measures as needed.",
                "Harvest Planning: Plan harvest and maintenance activities to maximize yield and plant health."
            ]
        }
    else:
        return {
            "class": "Dense Vegetation",
            "suggestion": [
                "Pruning and Thinning: If applicable, consider thinning crowded areas to allow better light penetration.",
                "Disease Management: Monitor for plant diseases and take preventive measures as needed.",
                "Harvest Planning: Plan harvest and maintenance activities to maximize yield and plant health."
            ]
        }  
class NDVI : 
    def train_ndvi_model(self):
        # Load your CSV data into a Pandas DataFrame
        data = pd.read_csv(r"C:\Users\farai\OneDrive\Desktop\Documents\Hounors\Artifact\CropSenseAPI\Backend\classifiers\Potch farm data.csv", sep=";")

        # Define NDVI categories based on the provided ranges
        def categorize_ndvi(ndvi_value):
            if ndvi_value <= 0.2:
                return 0  # Barren rock, sand, or snow
            elif 0.2 < ndvi_value <= 0.5:
                return 1  # Sparse vegetation
            elif 0.5 < ndvi_value <= 0.9:
                return 2  # Dense vegetation
            else:
                return -1  # Undefined

        data['NDVI_Category'] = data['NDVI'].apply(categorize_ndvi)

        # Split the data into features (X) and target (y)
        X = data[['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07', 'Month']]
        y = data['NDVI_Category']

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
    
   
        
    def predict_ndvi(self,bands,model):
        # Train the model (You can call this function once to train the model)
        # Load the pre-trained model
        # Ensure the input is a 2D array
       
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
        predicted_label = reverse_categorize_ndvi(predicted_categories)

        print(f'Predicted NDVI Category: {predicted_label}')

        return predicted_label
