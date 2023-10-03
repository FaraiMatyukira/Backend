import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
def reverse_categorize_msavi(category):
    if category == 0:
        return {
            "class": "Barren",
            "suggestion": [
                "Soil Testing: Conduct soil tests to assess nutrient deficiencies or pH imbalances.",
                "Soil Improvement: Add organic matter to improve soil quality.",
                "Consider Soil Erosion Control: Implement measures to prevent soil erosion in barren areas."
            ]
        }
    elif category == 1:
        return {
            "class": "Germination stage",
            "suggestion": [
                "Optimize Planting: Ensure proper planting depth and spacing for germinating crops.",
                "Water Management: Provide adequate moisture for seed germination.",
                "Monitor for Pests and Diseases: Be vigilant for early signs of pest or disease issues during this vulnerable stage."
            ]
        }
    elif category == 2:
        return {
            "class": "Germination stage",
            "suggestion": [
                "Optimize Planting: Ensure proper planting depth and spacing for germinating crops.",
                "Water Management: Provide adequate moisture for seed germination.",
                "Monitor for Pests and Diseases: Be vigilant for early signs of pest or disease issues during this vulnerable stage."
            ]
        }
    elif category == 3:
        return {
            "class": "Vegetation covers soil",
            "suggestion": [
                "Regular Maintenance: Continue with regular maintenance, including irrigation and fertilization.",
                "Harvest Planning: Plan for the upcoming harvest and post-harvest activities.",
                "Monitor for Overcrowding: Be mindful of plant spacing to avoid overcrowding and resource competition."
            ]
        }

    else:
        return {"class": "Unknown category", "suggestion": []}
class MSAVI:
    def train_msavi_model(self):
    # Load your data (replace 'data.csv' with the actual file name)
        data = pd.read_csv(r"C:\Users\farai\OneDrive\Desktop\Documents\Hounors\Artifact\CropSenseAPI\Backend\classifiers\Potch farm data.csv",sep =";")

        # Calculate MSAVI
        data['MSAVI'] = (2 * data['sur_refl_b02'] + 1 - np.sqrt((2 * data['sur_refl_b02'] + 1)**2 - 8 * (data['sur_refl_b02'] - data['sur_refl_b01']))) / 2


        def categorize_msavi(msavi_value):
            if msavi_value <= 0.2:
                return 0  # Barren rock, sand, or snow
            elif 0.2 < msavi_value <= 0.4:
                return 1  # Germination stage
            elif 0.4 < msavi_value <= 0.6:
                return 2  # Leaf development stage
            elif msavi_value  > 0.6:
                return 3 # Soil cover land

        data['MSAVI_Category'] = data['MSAVI'].apply(categorize_msavi)
        data['EVI']= data['EVI']/10000
        # Split the data into features (X) and target (y)
        X = data[['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07', 'Month']]
        y = data['MSAVI_Category']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert target to one-hot encoded labels
        y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=4)
        y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=4)
    
        # Build the ANN model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
            
        ])
        

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model for more epochs
        model.fit(X_train_scaled, y_train_encoded, epochs=50, batch_size=32, validation_split=0.2)
        return model
    
   
        
    def predict_msavi(self,bands,model):
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
        # Convert the predicted category index to MSAVI label
        predicted_label = reverse_categorize_msavi(predicted_categories)

        print(f'Predicted MSAVI Category: {predicted_label}')

        return predicted_label
        


