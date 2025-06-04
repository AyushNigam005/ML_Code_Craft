from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load encodings
with open('location_encoding.pkl', 'rb') as f:
    location_encoding = pickle.load(f)

with open('cuisines_encoding.pkl', 'rb') as f:
    cuisines_encoding = pickle.load(f)

features = [
    'Average_Cost', 'Minimum_Order', 'Votes', 'Reviews', 'Delivery_Time',
    'Cost_Per_Order', 'Engagement', 'Num_Cuisines',
    'Location_Encoded', 'Cuisines_Encoded'
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs from form
        avg_cost = float(request.form["Average_Cost"])
        min_order = float(request.form["Minimum_Order"])
        votes = float(request.form["Votes"])
        reviews = float(request.form["Reviews"])
        delivery_time = float(request.form["Delivery_Time"])
        cuisines = request.form["Cuisines"].strip()
        location = request.form["Location"].strip()

        # Feature engineering
        cost_per_order = avg_cost / min_order if min_order != 0 else 0
        engagement = votes * reviews
        num_cuisines = len(cuisines.split(',')) if cuisines else 0

        # Encode Location and Cuisines with fallback to global mean rating
        global_mean_rating = np.mean(list(location_encoding.values()))
        location_encoded = location_encoding.get(location, global_mean_rating)
        cuisines_encoded = cuisines_encoding.get(cuisines, global_mean_rating)

        # Create DataFrame for the model
        input_df = pd.DataFrame([[avg_cost, min_order, votes, reviews, delivery_time,
                                  cost_per_order, engagement, num_cuisines,
                                  location_encoded, cuisines_encoded]],
                                columns=features)

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        return render_template("index.html", prediction=round(prediction, 2))
    except Exception as e:
        return render_template("index.html", prediction="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
