import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



def get_model(file_name):
    try:
        # Load your CSV data into a Pandas DataFrame
        data = pd.read_csv(r"C:\Users\S_CSIS-PostGrad\Documents\Hounors\Research Projects\Artefact\Backend\predictors\data\Grobler Boerdery_ndvi_Area3.csv",sep = ";")
        # data = pd.read_csv(r"C:\Users\farai\OneDrive\Desktop\Documents\Hounors\Artifact\CropSenseAPI\Backend\predictors\data\Grobler Boerdery_ndvi_Area3.csv",sep = ";")

        data['MSAVI'] = (2 * data['sur_refl_b02'] + 1 - np.sqrt((2 * data['sur_refl_b02'] + 1)**2 - 8 * (data['sur_refl_b02'] - data['sur_refl_b01']))) / 2
        # Split the data into features (X) and target (y)
        X = data[[ 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07', 'Month']]
        y = data['MSAVI']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a linear regression model
        linear_model = LinearRegression()
        # Train the linear regression model
        linear_model.fit(X_train, y_train)

        
       
        return linear_model,X_test
    except Exception as e: 
        print("MSAVI error [get_model]",e)

# Convert predicted categories back to original NDVI labels
def categorize_msavi(msavi_value):
    if msavi_value <= 0.2:
        return 0  # Barren rock, sand, or snow
    elif 0.2 < msavi_value <= 0.4:
        return 1  # Germination stage
    elif 0.4 < msavi_value <= 0.6:
        return 2  # Leaf development stage
    elif msavi_value  > 0.6:
        return 3 # Soil cover land
    else:
        return -1  # Undefined
    

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

    
class MSAVI_predictions():

    def get_polygon_profile(self):
        try:
            # Create a linear regression model
            linear_model, X_test = get_model("jdfblsflasflds")

            # Make predictions using the linear regression model
            linear_predictions = linear_model.predict(X_test)


            # Calculate mean, max, and min of predictions
            mean_prediction = np.mean(linear_predictions)
            max_prediction = np.max(linear_predictions)
            min_prediction = np.min(linear_predictions)
            average=np.average(linear_predictions)

            payload  = {
               "mean_prediction": mean_prediction,
                "mean_prediction_category":reverse_categorize_msavi(categorize_msavi(mean_prediction)),

               "max_prediction": max_prediction,
               "max_prediction_category":reverse_categorize_msavi (categorize_msavi(max_prediction)),

               "min_prediction": min_prediction,
               "min_prediction_category": reverse_categorize_msavi (categorize_msavi(min_prediction)),

               "average":average,
               "average_category":reverse_categorize_msavi (categorize_msavi(average))

            }
            return payload
        except Exception as e  : 
            print("MSAVI [get_polygon_profile]")
            


    def get_forcast(self,list_of_data):
        try:
            linear_model, X_test = get_model("jdfblsflasflds")
            #  Define a test case for linear regression
            # linear_test_case = np.array([[643.06, 2550.2599999999998, 292.28999999999996, 1959.3499999999997, 3]]) format
            linear_test_case = np.array([list_of_data])
            # Make predictions using the linear regression model
            linear_prediction = linear_model.predict(linear_test_case)

            print(f"Linear Regression Prediction: {linear_prediction[0]}")
        except Exception as e : 
            print("MSAVI [get_forcast]")
