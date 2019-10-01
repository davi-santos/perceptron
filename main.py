import pandas as pd
import Perceptron as pc
import matplotlib.pyplot as plt
import numpy as np

def main():
    #read dataset from file
    df = pd.read_csv('iris.data', header=None)

    #select setosa e versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    #extract sepal length and pental length
    X = df.iloc[0:100, [0, 2]].values

    #plot data
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()