import numpy as np
import statsmodels.api as sm
import csv

def linear_regression(x, y):
    X = np.column_stack((x, 1/x))
    X = sm.add_constant(X)
    mod = sm.OLS(y, X)
    res = mod.fit()
    return res
#    print(res.summary())

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
    return data_dict

def preprocess(data_dict):
    new_group = [] 
    new_c1 = []
    new_c2 = []
    new_time_act = []
    flag = 0
    for i in list(range(len(data_dict['group']))):
        if data_dict['group'][i] == 1:
            new_group.append(data_dict['group'][flag:i])
            new_c1.append(data_dict['c1'][flag:i])
            new_c2.append(data_dict['c2'][flag:i])
            new_time_act.append(data_dict['time_act'][flag:i])
            flag = i
    new_group.append(data_dict['group'][flag:])
    new_c1.append(data_dict['c1'][flag:])
    new_c2.append(data_dict['c2'][flag:])
    new_time_act.append(data_dict['time_act'][flag:])
    data_dict.update({'group': new_group[1:]})
    data_dict.update({'c1': new_c1[1:]})
    data_dict.update({'c2': new_c2[1:]})
    data_dict.update({'time_act': new_time_act[1:]})
    return data_dict
    

def main():
    data_dict = read_csv('./V100-16GB.csv')
    data_dict = preprocess(data_dict)
    for i in list(range(len(data_dict['group']))):
#        print(data_dict['group'][i])
        res = linear_regression(data_dict['group'][i][:-2], data_dict['time_act'][i][:-2])
        if res.rsquared < 0.9:
            print(data_dict['c1'][i][0], data_dict['c2'][i][0])
            print('Parameters: ', res.params)
            print('R2: ', res.rsquared)

if __name__ == '__main__':
    main()
