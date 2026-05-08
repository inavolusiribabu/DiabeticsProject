from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pickle
import numpy as np

app = FastAPI()

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Home Page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": None}
    )


# Prediction
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Pregnancies: int = Form(...),
    Glucose: float = Form(...),
    BloodPressure: float = Form(...),
    SkinThickness: float = Form(...),
    Insulin: float = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: int = Form(...)
):

    features = np.array([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]])

    prediction = model.predict(features)[0]

    result = "Diabetes Positive 🩺" if prediction == 1 else "Diabetes Negative ✅"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": result
        }
    )