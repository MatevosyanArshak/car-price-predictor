import pandas as pd
from controllers.input_car_data import InputCarData
from controllers.data_cleaning_controller import CleanData
from controllers.train_test_split import ModelController

df = pd.read_csv('Data/vehicles-so.csv', encoding='latin1')

# TODO: Add better way to normalize data


data_clean_controller = CleanData(df)
df_col = ['odometer', 'price']
data_clean_controller.cleaning_sequence(df_col)
df = data_clean_controller.get_dataframe()

# please enter test size
traintest_split = ModelController(df)
traintest_split.train_sequence()
reg = traintest_split.get_model()
while True:
    car = InputCarData()
    car.get_data()

    print(reg.predict([[car.year, car.odometer]]))
