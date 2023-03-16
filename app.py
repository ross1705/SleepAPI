from flask import Flask, send_from_directory
from flask_restful import Api, Resource
from flask_cors import CORS
import pandas as pd
import pickle
import os

app = Flask(__name__)
CORS(app)
api = Api(app)

class SleepEfficiencyPrediction(Resource):
    def get(self, phone_usage_minutes):
        phone_usage_minutes = [int(phone_usage_minutes)]
        df = pd.DataFrame(phone_usage_minutes, columns=['night time phone usage / day (minutes)'])

        # Load the trained Random Forest model from the file
        with open("trained_model_rf.pkl", "rb") as file:
            rf_regressor = pickle.load(file)

        prediction = rf_regressor.predict(df)
        prediction = int(prediction[0])
        return str(prediction)

api.add_resource(SleepEfficiencyPrediction, '/prediction/<int:phone_usage_minutes>')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5019))
    app.run(debug=True, host='0.0.0.0', port=port)
