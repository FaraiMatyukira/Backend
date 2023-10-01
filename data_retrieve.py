import os
import pandas as pd
import numpy as np
from classifiers.Classifier_Service import farm_Classifier
from classifiers.ANN import NDVI
from classifiers.ANN2 import EVI
from classifiers.ANN3 import MSAVI

class retrieve : 
    def get_tail(self):
        try: 
            isinstance= NDVI()
            ndvi_model  = isinstance.train_ndvi_model()
            isinstance2= EVI()
            evi_model  = isinstance2.train_evi_model()

            isinstance3= MSAVI()
            msavi_model  = isinstance3.train_msavi_model()
            # folder_path = r"C:\Users\S_CSIS-PostGrad\Documents\Hounors\Research Projects\Artefact\Backend\predictors\data"
            folder_path = r"C:\Users\farai\OneDrive\Desktop\Documents\Hounors\Artifact\CropSenseAPI\Backend\predictors\data"
            filenames = os.listdir(folder_path)
            # Print the list of filenames
            subsection_files =[]
            for filename in filenames:
                subsection_files.append(filename)

            payload = []
            for file in subsection_files:
                # path  = f"C:/Users/S_CSIS-PostGrad/Documents/Hounors/Research Projects/Artefact/Backend/predictors/data/{str(file)}"
                path  = f"C:/Users/farai/OneDrive/Desktop/Documents/Hounors/Artifact/CropSenseAPI/Backend/predictors/data/{str(file)}"

                data = pd.read_csv(path,sep = ";")
                array =[]
               


                for i in range(1,5):
                    msavi_value  = ((2 * float(data.iloc[-i]['sur_refl_b02'])+ 1 - np.sqrt((2 * float(data.iloc[-i]['sur_refl_b02']) + 1)**2 - 8 * (float(data.iloc[-i]['sur_refl_b02']) - float(data.iloc[-i]['sur_refl_b01'])))) / 2)
                    value = {
                        "subsection_name": f"{str(file)}",
                        "timestamp": str(data.iloc[-i]["Timestamp"]),
                        "NDVI": str(data.iloc[-i]["NDVI"]),
                        "EVI": str(float(data.iloc[-i]["EVI"])/10000),
                        "MSAVI":str(msavi_value),
                        "bands":[float(data.iloc[-i]["sur_refl_b01"]),float(data.iloc[-i]["sur_refl_b02"]),float(data.iloc[-i]["sur_refl_b03"]),float(data.iloc[-i]["sur_refl_b07"]),int(data.iloc[-i]["Month"])],
                        "class":{
                            "ndvi_class": isinstance.predict_ndvi([float(data.iloc[-i]["sur_refl_b01"]),float(data.iloc[-i]["sur_refl_b02"]),float(data.iloc[-i]["sur_refl_b03"]),float(data.iloc[-i]["sur_refl_b07"]),int(data.iloc[-i]["Month"])],ndvi_model),
                            "evi_class": isinstance2.predict_evi([float(data.iloc[-i]["sur_refl_b01"]),float(data.iloc[-i]["sur_refl_b02"]),float(data.iloc[-i]["sur_refl_b03"]),float(data.iloc[-i]["sur_refl_b07"]),int(data.iloc[-i]["Month"])],evi_model),
                            "msavi_class":isinstance3.predict_msavi([float(data.iloc[-i]["sur_refl_b01"]),float(data.iloc[-i]["sur_refl_b02"]),float(data.iloc[-i]["sur_refl_b03"]),float(data.iloc[-i]["sur_refl_b07"]),int(data.iloc[-i]["Month"])],msavi_model)
                }
                        }

                    array.append(value)
                payload.append(array)
            
            return   payload
        except Exception as e : 
            print("error: data_retrieve=> ", e)

