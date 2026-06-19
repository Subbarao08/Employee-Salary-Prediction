import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load Dataset
df = pd.read_csv("C:/Employee-Salary-Prediction/dataset/Salary Data.csv")

# Remove missing values
df = df.dropna()

# Encode text columns
gender_encoder = LabelEncoder()
education_encoder = LabelEncoder()
job_encoder = LabelEncoder()

df["Gender"] = gender_encoder.fit_transform(df["Gender"])
df["Education Level"] = education_encoder.fit_transform(df["Education Level"])
df["Job Title"] = job_encoder.fit_transform(df["Job Title"])

# Features
X = df[
    [
        "Age",
        "Gender",
        "Education Level",
        "Job Title",
        "Years of Experience"
    ]
]

# Target
y = df["Salary"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Test Data
y_pred = model.predict(X_test)

# Evaluation
score = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Model Accuracy (R2 Score):", score)
print("Mean Absolute Error:", mae)

# Save Model
joblib.dump(
    model,
    "C:/Employee-Salary-Prediction/models/salary_model.pkl"
)

print("Model Saved Successfully!")

# Sample Prediction
sample_employee = [[
    30,   # Age
    1,    # Gender
    1,    # Education Level
    50,   # Job Title
    5     # Years of Experience
]]

predicted_salary = model.predict(sample_employee)

print("Predicted Salary:", predicted_salary[0])