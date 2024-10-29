from django.shortcuts import render
import pandas as pd
import pickle
import numpy as np
# Create your views here.

model = pickle.load(open('./model/LinearRegressionModelnewUpdated.pkl', 'rb'))
car = pd.read_csv('./Data/Cleaned_Car_data.csv')


def index(request):
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render(request, "index.html", {'companies': companies, 'car_models': car_models, 'year': year, 'fuel_type': fuel_type})


def predict(request):
    if request.method == "POST":
        company = request.POST.get('company')

        car_model = request.POST.get('car_models')
        year = request.POST.get('year')
        fuel_type = request.POST.get('fuel_type')
        driven = request.POST.get('kilo_driven')
        prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array(
            [car_model, company, year, driven, fuel_type]).reshape(1, 5)))
        print(prediction)
        result = str(np.round(prediction[0], 2))
        return render(request, "result.html", {'result': result})
