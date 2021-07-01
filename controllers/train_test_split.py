import pickle
from sklearn import linear_model
from sklearn.model_selection import train_test_split


class ModelController:

    def __init__(self, df, split_percentage=0.2):
        self.split_percentage = split_percentage
        self.df = df
        self.reg = linear_model.LinearRegression()
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def split(self):
        x = self.df[['year', 'odometer']]
        y = self.df['price']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.split_percentage, random_state=42)

    def fit_data(self):
        self.reg.fit(self.x_train, self.y_train)

    def train_sequence(self):
        self.split()
        self.read_model()


    def save_model(self):
        pickle_out = open("../Models/model.pickle", "wb")
        pickle.dump(self.reg, pickle_out)
        pickle_out.close()

    def read_model(self):
        try:
            pickle_in = open("../Models/model.pickle", "rb")
            self.reg = pickle.load(pickle_in)

        except FileNotFoundError:
            self.fit_data()
            self.save_model()

    def get_model(self):
        return self.reg
