from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load(
    "C:/Employee-Salary-Prediction/models/salary_model.pkl"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    gender = int(request.form["gender"])
    education = int(request.form["education"])
    job = int(request.form["job"])
    experience = float(request.form["experience"])

    data = pd.DataFrame(
        [[age, gender, education, job, experience]],
        columns=[
            "Age",
            "Gender",
            "Education Level",
            "Job Title",
            "Years of Experience"
        ]
    )

    prediction = model.predict(data)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Salary: ₹{prediction[0]:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)