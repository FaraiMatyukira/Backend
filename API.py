
from flask import Flask, request, json, jsonify,render_template
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from bson.json_util import dumps
from bson import json_util
from flask_cors import CORS
import json


app=Flask(__name__)

app.config["MONGO_URI"] = "mongodb://localhost:27017/Hermes"
def parse_json(data):
    return json.loads(json_util.dumps(data))

mongo  = PyMongo(app)
CORS(app)
@app.route("/User/signup",methods=["POST"])
def signup():
    status = 200
    resp  ={}
    try:
        data = request.get_json("data")
        print(data)
        username = data["data"]["username"]
        email = data["data"]["email"]
        password = data["data"]["password"]
        database_check = mongo.db.user.find_one({"username":f"{username}","email":f"{email}"})
        print(parse_json(database_check))
        if parse_json(database_check) == None:
            if username != "" and email !="" and password != "":

                payload = {
                    "username":username ,
                    "email":email,
                    "password":password
                }
                mongo.db.user.insert_one(payload) 
                status = 200  
                resp = {"message":"user made", "token":"0"}
        else:
            print("we being used ")
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
        username = data["data"]["username"]
        password = data["data"]["password"]
        if username != "" and password != "":
            database_check  = mongo.db.user.find_one({"username":f"{username}"})
            print(parse_json(database_check))
            if parse_json(database_check) != None:

                database_password  = database_check["password"]
                if password == database_password:
                    data= parse_json(database_check)
                    resp ={"message":"success","user":data,"token":"0"}
                else:
                    status =200
                    resp  ={"message":"User password is incorrect","token":"1"}
            else :
                status = 200
                resp = {"message":"User does not exsit","token":"1"}
        return jsonify(resp),status
    except Exception as e:
        print("ERROR on /User/Login",e)
        return jsonify(resp), status
    
app.route("/get/cordinates",methods=["POST"])
def get_cordinates():
    status = 200
    resp = {}
    try:
        data  = request.get_json("data")
    except Exception as e :
        print("ERROR on /get/cordinates: ",e)
        return jsonify(resp),status



if __name__  =="__main__":
    app.run(debug=True)
    # app.run(host='0.0.0.0',port=5000)