# gradProjectAPI

The dataset is based on historical sales by Favorita stores located in Ecuador from 1/1/2013 to 12/31/2016. Currently there is ongoing competition on Kaggle and 913 teams are competing on it. The dataset contains six csv’s files and 33 family categories (Eg. Automotive, Cleaning and seafood).
Due to the main idea of the project and for simplification, we have selected specific categories that would fit in our mobile application idea.
1- Automotive.
2- Cleaning.
3- Personal Care.
4- Bakery.
5- Frozen Food.
6- Hardware.
7- Seafood.

Various traditional machine learning models are used Eg. SARIMA, ARIMA and PROPHET to achieve the best results according to the evaluation metrics Eg. RMSE and MSE but LSTM which is based on RNN architecture (Neural Network) achieved the best results

Parameter tunning has been done to improve the performace of LSTM.


API: On the local machine, the model is trained and specified with each validation split to each correspondent category. Once training is complete, the models are saved to be used for predictions.
The web API is then deployed with the models for each category present on the server.
The web API contains a single GET request that receives two parameters “category” and “n_days” through a query string. Based on the parameters’ values the correct model is used to obtain a prediction for future demand. 
Predictions are returned in JSON format where each object consists of a date and corresponding sales.
