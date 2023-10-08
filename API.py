
from flask import Flask, request, json, jsonify,render_template
from flask_pymongo import PyMongo
from bson import json_util
from flask_cors import CORS
import datetime
import json
import jwt
import http.client
import bson
import uuid
import os 
# from data_gathering.gatherdataprofiles import Gather
from predictors.EVIprediction import EVI_predictions
from data_retrieve import retrieve
from predictors.NDVIprediction import NDVI_predictions
from predictors.MSAVIprediction import MSAVI_predictions
from classifiers.Classifier_Service import farm_Classifier


app=Flask(__name__)

app.config["MONGO_URI"] = "mongodb://localhost:27017/CropSense"
app.config["SECRET_KEY"] ="flow@master#alphaV1$%^#&@672"
def parse_json(data):
    return json.loads(json_util.dumps(data))

mongo  = PyMongo(app)
CORS(app)

@app.route ("/unprotected",methods= ["POST"])
def unprotected():
    status  = 200
    resp = {}
    try:
        resp   = {"message":"anyone can view this"}
    except Exception as e :
        status= 403
        resp={"message":f"{e}","status":"fail"}
        print("ERORR (/unprotected route)--->",e)
    return jsonify(resp),status
@app.route ("/protected",methods= ["POST"])
def protected():
    status  = 200
    resp = {}
    try:
        data = request.get_json("data")
        token = data["data"]["token"]
        if token == "":
            return jsonify({"message":"invalid"}),403
        try:
            data = jwt.decode(token, app.config["SECRET_KEY"],algorithms=['HS256'])
            
            resp  = {"message":"valid","email":data["email"]}
            return jsonify(resp),200
        except:
            return jsonify({"message":'invalid'}),403

    except Exception as e :
        status= 403
        resp={"message":f"{e}","status":"fail"}
        print("ERORR (/protected route)--->",e)
    return jsonify(resp),status

@app.route("/User/signup",methods=["POST"])
def signup():
    status = 200
    resp  ={}
    try:
        data = request.get_json("data")
        print(data)
        email = data["data"]["email"]
        password = data["data"]["password"]
        lat = data["data"]["lat"]
        lng = data["data"]["lng"]
        database_check = mongo.db.user.find_one({"email":f"{email}"})
        if parse_json(database_check) == None:
            if  email !="" and password != "":

                payload = {
                    "email":f"{email}",
                    "password":f"{password}",
                    "user_id":1,
                    "lat":lat,
                    "lng":lng
                }
                mongo.db.user.insert_one(payload) 
                status = 200  
                token = jwt.encode({"email":email,"exp":datetime.datetime.utcnow()+ datetime.timedelta(minutes=180)},app.config["SECRET_KEY"])
                resp = {"message":"User account made", "token":token,'status':status}
                print(resp)
                return jsonify(resp),status
        else:
            status = 200
            resp ={"message":"User credentials are in use","token":"1"}
        return jsonify(resp),status 
    except Exception as e  :
        print("ERROR on /User/signup",e)
        return jsonify(resp), status
    
@app.route("/User/Login",methods=["POST"])
def login():
    status =200
    resp = {}
    try:
        data = request.get_json("data")
        print(data)
        email = data["data"]["email"]
        password = data["data"]["password"]
        if email != "" and password != "":
            database_check  = mongo.db.user.find_one({"email":f"{email}"})
            print(parse_json(database_check))
            if parse_json(database_check) != None:
                print(password)
                database_password  = database_check["password"]
                if password == database_password:
                    print(" they are the same")
                    token = jwt.encode({"user_number":email,"exp":datetime.datetime.utcnow()+ datetime.timedelta(minutes=180)},app.config["SECRET_KEY"])
                    data= parse_json(database_check)
                    resp ={"message":"success","user":data,"token":token,"status":status}
                    print(resp)
                else:
                    status =400
                    resp  ={"message":"User password is incorrect"}
            else :
                status = 200
                resp = {"message":"User does not exsit","token":"1"}
        return jsonify(resp),status
    except Exception as e:
        print("ERROR on /User/Login",e)
        return jsonify(resp), status
@app.route("/get/center",methods=["GET"])
def get_center():
    status =200
    resp = {}
    try:
      
        database_check  = parse_json(mongo.db.user.find_one({"user_id":"1"}))
        payload ={
            "lat":float(database_check["lat"]),
            "lng":float(database_check["lng"])
        }
        return jsonify(payload),status
    except Exception as e:
        print("ERROR on /get/center",e)
        return jsonify(resp), status  
@app.route("/set/corodinates/dataprofile", methods=["POST"])
def set_profile_data():
    status  = 200
    resp  = {}
    try: 
        data = request.get_json("data")
        print(data)
        if data != "":
            payload  = {
                "email": data["data"]["email"],
                "coordinates":data["data"]["coordinates"],
                "subsection_name": data["data"]["subsection_name"],
                "file_name": data["data"]["file_name"]
            }
            print(payload)
            response = mongo.db.coordinates.insert_one(payload) 
            resp = {"message": "Profile coordinates saved"}
            return jsonify(resp), status
    except Exception  as e : 
        print("ERROR on /set/corodinates/dataprofile",e)
        return jsonify(resp), status
@app.route("/get/data/file", methods=["POST"])
def get_data():
    status  = 200
    resp  = {}
    try: 
        file1 = request.files["file"]
    
        # Check if the file is present in the request
        if file1:
            # Specify the folder where you want to save the file
            upload_folder = "predictors/data"

            # If the folder doesn't exist, create it
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            # Save the file to the specified folder
            file_path = os.path.join(upload_folder, file1.filename)
            file1.save(file_path)

            resp["message"] = "File successfully saved."
        else:
            resp["message"] = "No file provided in the request."

        return jsonify(resp), status
        return jsonify(resp), status
    except Exception  as e : 
        print("ERROR on /get/data/file",e)
        return jsonify(resp), status
@app.route("/delete/corodinates/dataprofile", methods=["POST"])
def delete_profile_data():
    status  = 200
    resp  = {}
    try: 
        data = request.get_json("data")
        name = data["data"]["subsection_name"]
        print(data)
        if data != "":
            response_file = parse_json(mongo.db.coordinates.find({"subsection_name":name}))
            print(response_file)
            if response_file != []:
                    filename = response_file[0]["file_name"]
                    print(filename)
                    # Specify the folder where the file is located
                    upload_folder = "predictors/data"

                    # Create the full path to the file
                    file_path = os.path.join(upload_folder, filename)

                    # Check if the file exists before attempting to delete it
                    if os.path.exists(file_path):
                        # Delete the file
                        os.remove(file_path)
                        response = mongo.db.coordinates.delete_one({"subsection_name":name}) 
                        resp["message"] = f"Section coordinates deleted"
                    else:
                        resp["message"] = f"Section coordinates not deleted."
            
            resp = {"message": "Profile coordinates deleted"}
            return jsonify(resp), status
    except Exception  as e : 
        print("ERROR on /delete/corodinates/dataprofile",e)
        return jsonify(resp), status
@app.route("/get/corodinates/dataprofile",methods=["GET"])
def get_profile_data():
    status= 200
    resp  = {}
    try:
        response =mongo.db.coordinates.find({})
        data = parse_json(response)
        subsection  = []
        for i  in  data :
            payload  = {
               "coordinates":i["coordinates"],
                "subsection_name": i["subsection_name"] 
            }
            subsection.append(payload)
        resp = {"message":"Profile coordinates retrieved","data":subsection}
        return jsonify(resp), status
    except Exception as e:
        print("ERROR on /get/corodinates/dataprofile",e)
        return jsonify(resp), status
@app.route("/get/model/dataprofile",methods=["GET"])
def get_model_data():
    status= 200
    resp  = {}
    try:
        insatnce  = EVI_predictions()
        evi_data  = insatnce.get_polygon_profile()

        insatnce2  = NDVI_predictions()
        ndvi_data  = insatnce2.get_polygon_profile()

        insatnce3  = MSAVI_predictions()
        msavi_data  = insatnce3.get_polygon_profile()


        payload  = {
            "NDVI_profile":ndvi_data,
            "EVI_profile":evi_data,
            "MSAVI_profile":msavi_data

        }
        # response =mongo.db.coordinates.find({})
        # data = parse_json(response)
        # subsection  = []
        # for i  in  data :
        #     payload  = {
        #        "coordinates":i["coordinates"],
        #         "subsection_name": i["subsection_name"] 
        #     }
        #     subsection.append(payload)
        resp = {"message":"Profile coordinates retrieved","data":payload}
        return jsonify(resp), status
    except Exception as e:
        print("ERROR on /get/corodinates/dataprofile",e)
        return jsonify(resp), status
@app.route("/get/recent/bands",methods= ["GET"])
def get_recent_bands():
    status= 200
    resp  = {}
    try:
        print("hello")
        database_check  = parse_json(mongo.db.recentDataClass.find({}))
        if database_check ==[]:
            instance= retrieve()
            data  = instance.get_tail()
            payload = {
                "data":data
            }
            mongo.db.recentDataClass.insert_one({"data":data,"number":"1",})
            return jsonify(payload),status  
        else:
            payload = {
                "data":database_check[0]["data"]
            }
            return jsonify(payload),status  
        
    except Exception as e : 
        print("ERROR on /get/model/classes",e)
        return jsonify(resp), status
@app.route("/delete/recent/bands",methods= ["GET"])
def delete_recent_bands():
    status= 200
    resp  = {}
    try:
        database_check  = parse_json(mongo.db.recentDataClass.find({}))
        if database_check !=[]:
            print(database_check)
            response = mongo.db.recentDataClass.delete_one({"number":"1"})
            payload ={
                "message": "Recend bands deleted"
            }
            print(payload)
            return jsonify(payload),status  
        else:
            payload ={
                "message": "No recend bands"
            }
            return jsonify(payload),status 

    except Exception as e : 
        print("ERROR on /delete/recent/bands",e)
        return jsonify(resp), status
    
@app.route("/post/model/classes",methods= ["POST"])
def post_model_classes():
    status= 200
    resp  = {}
    try:
        data  = request.get_json("data")
        bands = data["data"]["bands"]
        if bands  != []:
            print("array recieved==>",bands)
            isinstance= farm_Classifier()
            payload  = {
                "ndvi_class": isinstance.NDVI(bands),
                "evi_class": isinstance.EVI(bands),
                "msavi_class":isinstance.MSAVI(bands)
            }
            print(payload)
            return jsonify(payload),status  
        else : 
            status = 400
            resp = {"message":"missing inputs"}
            return jsonify(payload),status 
    except Exception as e : 
        print("ERROR on /get/model/classes",e)
        return jsonify(resp), status
@app.route("/get/cordinates/weather",methods=["POST"])
def get_cordinates():
    status = 200
    resp = {}
    try:
        print("hello")
        data  = request.get_json("data")
        name = data["data"]["name"]
        conn = http.client.HTTPSConnection("ai-weather-by-meteosource.p.rapidapi.com")

        headers = {
            'X-RapidAPI-Key': "b97746ddd1mshc2043dd274eda0fp16cb26jsn4c213461107f",
            'X-RapidAPI-Host': "ai-weather-by-meteosource.p.rapidapi.com"
        }

        conn.request("GET", f"/find_places?text={name}&language=en", headers=headers)
        res = conn.getresponse()
        data = res.read()

        print(data.decode("utf-8"))
        # Assuming 'data' is the decoded byte data
        decoded_data = data.decode("utf-8")

        # Convert the decoded data to a JSON object
        json_data = json.loads(decoded_data)
        return jsonify(json_data),status
    except Exception as e :
        print("ERROR on /get/cordinates: ",e)
        return jsonify(resp),status

@app.route("/get/current/weather", methods = ["GET"])
def get_weather():
    status  = 200
    resp= {}
    try:
        conn = http.client.HTTPSConnection("ai-weather-by-meteosource.p.rapidapi.com")
        headers = {
            'X-RapidAPI-Key': "b97746ddd1mshc2043dd274eda0fp16cb26jsn4c213461107f",
            'X-RapidAPI-Host': "ai-weather-by-meteosource.p.rapidapi.com"
        }
        conn.request("GET", "/current?lat=26.67066&lon=27.08154&timezone=auto&language=en&units=auto", headers=headers)

        res = conn.getresponse()
        data = res.read()

        # Assuming 'data' is the decoded byte data
        decoded_data = data.decode("utf-8")

        # Convert the decoded data to a JSON object
        json_data = json.loads(decoded_data)
        return jsonify(json_data),status
    except Exception as e : 
        print("ERROR on /get/weather: ",e)
    return jsonify(resp),status

@app.route("/get/weather/alerts", methods = ["GET"])
def get_weather_alerts():
    status  = 200
    resp= {}
    try:
        conn = http.client.HTTPSConnection("ai-weather-by-meteosource.p.rapidapi.com")

        headers = {
            'X-RapidAPI-Key': "b97746ddd1mshc2043dd274eda0fp16cb26jsn4c213461107f",
            'X-RapidAPI-Host': "ai-weather-by-meteosource.p.rapidapi.com"
        }

        conn.request("GET", "/alerts?lat=45.74846&lon=4.84671&timezone=auto&language=en", headers=headers)

        res = conn.getresponse()
        data = res.read()

        print(data.decode("utf-8"))
    except Exception as e : 
        print("ERROR on /get/weather: ",e)
    return jsonify(resp),status

if __name__  =="__main__":
    app.run(debug=True)