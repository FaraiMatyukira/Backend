import os
import pandas as pd
import numpy as np
class retrieve : 
    def get_tail(self):
        try: 
            folder_path = r"C:\Users\S_CSIS-PostGrad\Documents\Hounors\Research Projects\Artefact\Backend\predictors\data"
            filenames = os.listdir(folder_path)
            # Print the list of filenames
            subsection_files =[]
            for filename in filenames:
                subsection_files.append(filename)

            payload = []
            for file in subsection_files:
                path  = f"C:/Users/S_CSIS-PostGrad/Documents/Hounors/Research Projects/Artefact/Backend/predictors/data/{str(file)}"
                data = pd.read_csv(path,sep = ";")
                array =[]
                for i in range(1,5):
                    value = {
                        "subsection_name": f"{str(file)}",
                        "timestamp": str(data.iloc[-i]["Timestamp"]),
                        "bands":[float(data.iloc[-i]["sur_refl_b01"]),float(data.iloc[-i]["sur_refl_b02"]),float(data.iloc[-i]["sur_refl_b03"]),float(data.iloc[-i]["sur_refl_b07"]),float(data.iloc[-i]["Month"])]}
                    array.append(value)
                payload.append(array)
            return   payload
        except Exception as e : 
            print("error: data_retrieve=> ", e)
