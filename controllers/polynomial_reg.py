from sklearn.preprocessing import PolynomialFeatures

class PolynomialReg:

    def __init__(self, df):
        self.df = df
        self.poly_reg = PolynomialFeatures(degree=2)
