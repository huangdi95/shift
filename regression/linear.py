import numpy as np
import statsmodels.api as sm
import csv
import matplotlib.pyplot as plt

def linear_regression(y, X):
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
    
def figplt(data_x, data_y, params, mode='nike', start=0.1, end=512):
    x = np.linspace(start, end, 1000, endpoint=True)
    if mode == 'nike':
        a = params[1]
        b = params[2]
        c = params[0]
        y = a / x + b * x + c
    elif mode == 'linear':
        a = params[1]
        c = params[0]
        y = a * x + c
    plt.plot(x, y)
    plt.scatter(data_x, data_y)

def group_regression(data_dict, show_plot=False):
    group_params_list = []
    labels = []
    for i in list(range(len(data_dict['group']))):
        labels.append(str([int(data_dict['c1'][i][0]), int(data_dict['c2'][i][0])]))
#        print(data_dict['group'][i])

        x = data_dict['group'][i][:-2]
        y = data_dict['time_act'][i][:-2]

        X = np.column_stack((1/x, x))
        res = linear_regression(y, X)
        if res.rsquared < 0.9:
            print(data_dict['c1'][i][0], data_dict['c2'][i][0])
            print('Parameters: ', res.params)
            print('R2: ', res.rsquared)
        group_params_list.append(res.params)
        if show_plot:
            figplt(x, y, res.params)
    if show_plot:
        plt.legend(labels=labels)
        plt.show()
    return group_params_list 

def channel_regression(data_dict, group_params_list, show_plot=False):
    y = [] 
    c1 = []
    c2 = []
    for i in list(range(len(group_params_list))):
        c1.append(data_dict['c1'][i][0])
        c2.append(data_dict['c2'][i][0])
        y.append(group_params_list[i][1])
    c1 = np.array(c1)
    c2 = np.array(c2)
    y = np.array(y)
    X = c1 * c2
    res = linear_regression(y, X)
    print('Parameters: ', res.params)
    print('R2: ', res.rsquared)
    if show_plot:
        figplt(X, y, res.params, 'linear', 0, 7e6)
        plt.legend(labels='channel')
        plt.show()

def main():
    data_dict = read_csv('./V100-16GB.csv')
    data_dict = preprocess(data_dict)
    group_result_list = group_regression(data_dict)
    channel_regression(data_dict, group_result_list, True)

if __name__ == '__main__':
    main()
