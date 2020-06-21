import numpy as np
import pandas as pd
from sklearn import datasets


def normal_distribution(x, mean, covariance):
    "pdf of the multivariate normal distribution."
    d = 2
    x_m = x - np.squeeze(np.asarray(mean))
    return (1.0 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance))) *
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))


def covariance_matrix(dataset, mean, k, clusternumber):
    "Calculates the covariance matrix of each cluster"
    n = dataset.shape[0]
    cov_mat = np.zeros(shape=(k, k))
    pcluster2 = dataset.loc[:, 'Cluster2'].mean()
    pcluster1 = dataset.loc[:, 'Cluster1'].mean()
    for i in range(k):
        for j in range(k):
            for l in range(n):
                pcluster = 0
                if clusternumber == 1:
                    pdata = dataset.loc[l, 'Cluster1']
                    pcluster = pcluster1
                else:
                    pdata = dataset.loc[l, 'Cluster2']
                    pcluster = pcluster2
                cov_mat[i, j] += (pdata) * (dataset.loc[l, i] - mean[:, i]) * (dataset.loc[l, j] - mean[:, j]) / (
                        n * pcluster)
    return cov_mat


def mean_array(dataset, k, clusternumber):
    "Calculates the mean array of each cluster"
    mean = np.zeros(shape=(1, k))
    for i in range(k):
        for j in range(dataset.shape[0]):
            if clusternumber == 1:
                mean[:, i] += dataset.loc[j, 'Cluster1'] * dataset.loc[j, i]
            else:
                mean[:, i] += dataset.loc[j, 'Cluster2'] * dataset.loc[j, i]
        mean[:, i] = mean[:, i] / dataset.shape[0]
    return mean


def distancefunction(element1, element2, n):
    "Calculates the euclidean distance between two elements"
    distance = 0
    for i in range(n):
        distance += (element1.loc[i, 'Cluster1'] - element2.loc[i, 'Cluster1']) ** 2
    return distance


iris = datasets.load_iris()
x = iris.data
dataset = pd.DataFrame(x)  # Train dataset
pd.set_option('display.max_rows', dataset.shape[0] + 1)
k = 4  # Dimension

for i in range(dataset.shape[0]):  # Random probability for the first round
    dataset.loc[i, 'Cluster1'] = np.random.rand()
    dataset.loc[i, 'Cluster2'] = 1 - dataset.loc[i, 'Cluster1']

while True:
    olddataset = pd.DataFrame(dataset)
    cluster1_mean = mean_array(dataset, k, 1)
    cluster2_mean = mean_array(dataset, k, 2)
    cluster1_covariance = covariance_matrix(dataset, cluster1_mean, k, 1)
    cluster2_covariance = covariance_matrix(dataset, cluster2_mean, k, 1)

    for i in range(dataset.shape[0]):
        x0 = dataset.loc[i, 0]
        x1 = dataset.loc[i, 1]
        x2 = dataset.loc[i, 2]
        x3 = dataset.loc[i, 3]
        list = [x0, x1, x2, x3]
        x = np.array(list)
        pconditional_cluster1 = normal_distribution(x, cluster1_mean, cluster1_covariance) * dataset[
            'Cluster1'].mean()
        pconditional_cluster2 = normal_distribution(x, cluster2_mean, cluster2_covariance) * dataset[
            'Cluster2'].mean()
        dataset.loc[i, 'Cluster1'] = pconditional_cluster1 / (pconditional_cluster1 + pconditional_cluster2)
        dataset.loc[i, 'Cluster2'] = 1 - dataset.loc[i, 'Cluster1']

    if distancefunction(dataset, olddataset, dataset.shape[0]) < 0.01:
        break

print("The final clustering is:")
print(dataset)
