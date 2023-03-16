from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from flask import render_template
from flask import Flask, request, jsonify, render_template, send_from_directory


app = Flask(__name__)
CORS(app)
api = Api(app)

class SleepEfficiencyPrediction(Resource):
    def get(self, phone_usage_minutes):
        phone_usage_minutes = [int(phone_usage_minutes)]
        df = pd.DataFrame(phone_usage_minutes, columns=['night time phone usage / day (minutes)'])

        # Load the trained model from the file
        with open("trained_model.pkl", "rb") as file:
            lr = pickle.load(file)

        prediction = lr.predict(df)
        prediction = int(prediction[0])
        return str(prediction)

api.add_resource(SleepEfficiencyPrediction, '/prediction/<int:phone_usage_minutes>')

@app.route('/')
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5018)
