import os
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
from keras.callbacks import EarlyStopping
import tensorflow
from pandas.tseries.offsets import DateOffset

import json
import warnings
warnings.filterwarnings("ignore")
app = FastAPI()


@app.get('/')
def index():
    return 'hello world'



@app.get('/predict')
def index(n_days: int, category: str):
    sc = MinMaxScaler(feature_range=(-1, 1))
    new_df = pd.read_csv("{category}.csv".format(category=category))
    new_df_notDrop = new_df
    new_df = new_df.dropna()
    K = len(new_df)
    train = new_df.iloc[:, 1:2].values
    train_scaled = sc.fit_transform(train)

    if category == "automotive":
        history = load_model('automotive')
    elif category == "cleaning":
        history = load_model('cleaning')
    elif category == "personalcare":
        history = load_model('personalcare')
    elif category == "bakery":
        history = load_model('bakery')
    elif category == "frozenfood":
        history = load_model('frozenfood')
    elif category == "hardware":
        history = load_model('hardware')
    elif category == "seafood":
        history = load_model('seafood')

    WS =5
    prediction_test = []
    Batch_One = train_scaled[-WS:]
    Batch_New = Batch_One.reshape((1, WS, 1))

    for i in range(n_days):
        First_pred = history.predict(Batch_New)[0]
        prediction_test.append(First_pred)
        Batch_New = np.append(Batch_New[:, 1:, :], [[First_pred]], axis=1)

    prediction_test = np.array(prediction_test)
    prediction = sc.inverse_transform(prediction_test)
    # Creating New Dates
    last_date = new_df_notDrop.iloc[K - 1:, 0]
    pred = pd.read_csv("datetest.csv")
    pred["date"] = last_date

    pred.index = pd.to_datetime(pred['date'])


    pred_date = [pred["date"].index[-1] + DateOffset(days=x) for x in range(0, n_days + 1)]
    pred_date = pd.DataFrame(index=pred_date[1:], columns=["date","sales"])

    pred_date.index.name = 'date'
    pred_date_df = pred_date.index
    pred_date_df = pd.DataFrame(pred_date_df)
    pred_date_df["sales"] = prediction

    # if os.path.exists("prediction.csv"):
    #     os.remove("prediction.csv")
    pred_date_df.to_csv("prediction.csv", index=False)

    pred_date_df['date'] = pred_date_df['date'].astype(str)
    pred_date_df_list = pred_date_df.values.tolist()
    jsonStr = json.dumps(pred_date_df_list)



    return jsonStr



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)