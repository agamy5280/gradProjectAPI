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



@app.post("/",status_code=201)
async def PostFiles(uploadedFiles: list[UploadFile],category: str):
    df_list=[]
    for uploadedFile in uploadedFiles:
        if os.path.exists(uploadedFile.filename):
            os.remove(uploadedFile.filename)
        df= pd.read_csv(uploadedFile.file)
        df=df.fillna('')
        df_list.append(df.to_numpy())

    result = np.concatenate(df_list)
    result = pd.DataFrame(result)
    result.columns=["date","sales"]
    if os.path.exists("{category}.csv".format(category=category)):
        os.remove("{category}.csv".format(category=category))

    result.to_csv("{category}.csv".format(category=category), index=False)
    app.K = len(result)
    return df_list


@app.post('/train')
def index(category: str):
    if category == "automotive":
        validation_split_size = 0.1
    elif category == "cleaning":
        validation_split_size = 0.01
    elif category == "personalcare":
        validation_split_size = 0.1
    elif category == "bakery":
        validation_split_size = 0.05
    elif category == "frozenfood":
        validation_split_size = 0.01
    elif category == "hardware":
        validation_split_size = 0.01
    elif category == "seafood":
        validation_split_size = 0.1

    new_df = pd.read_csv("{category}.csv".format(category=category))
    new_df_notDrop = new_df
    new_df = new_df.dropna()
    # Preprocessing on whole dataset (train is whole dataset values)
    train = new_df.iloc[:, 1:2].values

    # Normalization
    sc = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = sc.fit_transform(train)

    # Creating empty sets and identifying window size
    x_train = []
    y_train = []
    WS = 5

    # extract values for each change sesonality to predict next value
    for i in range(WS, len(train_scaled)):
        x_train.append(train_scaled[i - WS:i, 0:1])
        y_train.append(train_scaled[i, 0])
    # converting list to array
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Model Building
    model = Sequential()
    model.add(LSTM(60, return_sequences=True, input_shape=(5, 1)))
    model.add(LSTM(60))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=20, patience=10)
    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=validation_split_size,callbacks=[es])

    if category == "automotive":
        model.save('automotive')
    elif category == "cleaning":
        model.save('cleaning')
    elif category == "personalcare":
        model.save('personalcare')
    elif category == "bakery":
        model.save('bakery')
    elif category == "frozenfood":
        model.save('frozenfood')
    elif category == "hardware":
        model.save('hardware')
    elif category == "seafood":
        model.save('seafood')


    return "train completed"

@app.get('/predict')
def index(n_days: int, category: str):
    sc = MinMaxScaler(feature_range=(-1, 1))
    new_df = pd.read_csv("{category}.csv".format(category=category))
    new_df_notDrop = new_df
    new_df = new_df.dropna()
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
    last_date = new_df_notDrop.iloc[app.K - 1:, 0]
    pred = pd.read_csv("datetest.csv")
    pred["date"] = last_date

    pred.index = pd.to_datetime(pred['date'])
   # df = pd.read_csv("train.csv")

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