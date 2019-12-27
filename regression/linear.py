import numpy as np
import statsmodels.api as sm
import csv

def linear_regression(x, y):
    spector_data = sm.datasets.spector.load(as_pandas=False)
    spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)
    x = np.array([1, 2, 3, 4, 5])
    X = np.column_stack((x, x**2))
    X = sm.add_constant(X)
    print(X)
    print(X.shape)
    beta = np.array([1, 0.1, 10])
    #y = np.dot(X, beta)
    y = np.array([4, 6, 4, 5, 5])
    print(y)
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(res.summary())
    print('Parameters: ', res.params)
    print('R2: ', res.rsquared)

def read_csv(filename):
    data = []
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader:
            data.append(row[1:])
    data = [[float(x) for x in row] for row in data]
    data = np.array(data)
    header = np.array(header)
    data_dict = {}
    for i in list(range(len(header[1:]))):
        data_dict.update({header[i+1]: data[:, i]})
    print(data_dict['c1'])
    print(data_dict['c2'])
    return data_dict

def preprocess(data_dict):
    pass  
    

def main():
    data_dict = read_csv('./V100-16GB.csv')

if __name__ == '__main__':
    main()
