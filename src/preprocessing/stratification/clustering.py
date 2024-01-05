# Import packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Stratification
def choose_K(df): # Choose number of pincipal components
    scaler = StandardScaler()
    df_scl = scaler.fit_transform(df)
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=20, max_iter=300,
            tol=1e-04, random_state=422
        )
        km.fit(df_scl)
        distortions.append(km.inertia_)

    # plot
    font = {'weight' : 'bold','size': 10}

    plt.rc('font', **font)
    k_c = plt.figure(figsize=(10, 6))
    plt.plot([str(i+1) for i in range(10)], distortions, marker='o')
    plt.xlabel('Number of clusters', fontsize=12)
    plt.ylabel('Distortion', fontsize=12)
    plt.show()
    k_c.savefig('k_means_clustering.png')

if __name__ == '__main__':
    print("")
    print("")
    print("|============================================================================|")
    print("|                                                                            |")
    print("|         -----                    STRATIFICATION             -----          |")
    print("|                                                                            |")
    print("|============================================================================|")
    print("")
    print("")
    print("********************************* INPUT DATA *********************************")
    print("")
    print("Import data may take several minutes, please wait...")
    print("")
    # Read genotype-phenotype data after subsequent data preprocessing
    X_train_init = pd.read_csv('../data/X_train.csv').set_index('sample')
    y_train = pd.read_csv('../data/y_train.csv').replace([1,2], [0, 1])['Phenotype']
    X_test_init = pd.read_csv('../data/X_test.csv').set_index('sample')
    y_test = pd.read_csv('../data/y_test.csv').replace([1,2], [0, 1])['Phenotype']
    choose_K(X_train_init.iloc[:, 0:-1])