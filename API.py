
from flask import Flask, request, json, jsonify,render_template
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from bson.json_util import dumps
from bson import json_util
from flask_cors import CORS
import datetime
import json
import jwt
import http.client
import bson
import uuid
# from data_gathering.gatherdataprofiles import Gather
from predictors.EVIprediction import EVI_predictions

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
        database_check = mongo.db.user.find_one({"email":f"{email}"})
        id = bson.Binary.from_uuid(uuid.uuid1())
        if parse_json(database_check) == None:
            if  email !="" and password != "":

                payload = {
                    "email":f"{email}",
                    "password":f"{password}",
                    "user_number":id
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
                "subsection_name": data["data"]["subsection_name"]
            }
            print(payload)
            response = mongo.db.coordinates.insert_one(payload) 
            resp = {"message": "Profile coordinates saved"}
            return jsonify(resp), status
    except Exception  as e : 
        print("ERROR on /set/corodinates/dataprofile",e)
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
        data  = insatnce.get_polygon_profile()
        # response =mongo.db.coordinates.find({})
        # data = parse_json(response)
        # subsection  = []
        # for i  in  data :
        #     payload  = {
        #        "coordinates":i["coordinates"],
        #         "subsection_name": i["subsection_name"] 
        #     }
        #     subsection.append(payload)
        resp = {"message":"Profile coordinates retrieved","data":data}
        return jsonify(resp), status
    except Exception as e:
        print("ERROR on /get/corodinates/dataprofile",e)
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
        conn.request("GET", "/current?lat=37.81021&lon=-122.42282&timezone=auto&language=en&units=auto", headers=headers)

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