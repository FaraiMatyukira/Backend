import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



# Convert predicted categories back to original NDVI labels
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
    
class farm_Classifier():
    def MSAVI(self,array):
   
        # loaded_model = joblib.load(r'C:\Users\S_CSIS-PostGrad\Documents\Hounors\Research Projects\Artefact\Backend\classifiers\MSAVI.joblib')
        loaded_model = joblib.load(r'C:\Users\farai\OneDrive\Desktop\Documents\Hounors\Artifact\CropSenseAPI\Backend\classifiers\MSAVI.joblib')

        new_data = np.array([array])
        # Standardize the test case using the same scaler used for training
        scaler = StandardScaler()
        scaler.fit(new_data)
        test_case_scaled = scaler.transform(new_data)

        # Make predictions
        predicted_category = loaded_model.predict(test_case_scaled)
        predicted_category_index = np.argmax(predicted_category)
        

        # Convert the predicted category index to NDVI label
        print(predicted_category_index)
        predicted_label = reverse_categorize_msavi(predicted_category_index)
        # print( "msavi",predicted_label)
        return predicted_label
  


    def NDVI(self,array):
        # loaded_model = joblib.load(r'C:\Users\S_CSIS-PostGrad\Documents\Hounors\Research Projects\Artefact\Backend\classifiers\NDVI.joblib')
        loaded_model = joblib.load(r'C:\Users\farai\OneDrive\Desktop\Documents\Hounors\Artifact\CropSenseAPI\Backend\classifiers\NDVI.joblib')

        # data = pd.read_csv(r"C:\Users\S_CSIS-PostGrad\Documents\Hounors\Research Projects\Artefact\Backend\predictors\data\Grobler Boerdery_ndvi_Area3.csv",sep = ";")
        # data['EVI']= data['EVI']/10000
        new_data = np.array([array])
        # Standardize the test case using the same scaler used for training
        scaler = StandardScaler()
        scaler.fit(new_data)
        test_case_scaled = scaler.transform(new_data)

        # Make predictions
        predicted_category = loaded_model.predict(test_case_scaled)
        print("predicted_category",predicted_category)
        predicted_category_index = np.argmax(predicted_category)
    
        # Convert the predicted category index to NDVI label
        print("predicted_category_index",predicted_category_index)
        predicted_label = reverse_categorize_ndvi(predicted_category_index)
        print( "ndvi",predicted_label)
        return predicted_label
       


    def EVI(self,array):
        # loaded_model = joblib.load(r'C:\Users\S_CSIS-PostGrad\Documents\Hounors\Research Projects\Artefact\Backend\classifiers\EVI.joblib')
        loaded_model = joblib.load(r'C:\Users\farai\OneDrive\Desktop\Documents\Hounors\Artifact\CropSenseAPI\Backend\classifiers\EVI.joblib')
        # Extract the features for prediction
        new_data = np.array([array])
        # Standardize the test case using the same scaler used for training
        scaler = StandardScaler()
        scaler.fit(new_data)
        test_case_scaled = scaler.transform(new_data)

        # Make predictions
        predicted_category = loaded_model.predict(test_case_scaled)
        predicted_category_index = np.argmax(predicted_category)
        print(predicted_category_index)

        # Convert the predicted category index to NDVI label
        predicted_label = reverse_categorize_evi(predicted_category_index)  
        return predicted_label
     
