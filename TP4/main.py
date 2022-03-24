import functions
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import dml

N = 5

def main():
    classes = [
        './classes/dolphin',
        './classes/flamingo',
        './classes/panda',
        './classes/wild_cat'
    ]

    # clusters = [2 ** x for x in range(1, 15)]
    clusters = [x for x in range(2, N+1)]
    print(clusters)

    variances = list()
    errors = list()

    for cluster in clusters:
        print(f'start for N={cluster}')
        variance, error, model = functions.vocabulaire(cluster, classes)
        variances.append(variance)
        errors.append(error)

    
    fig, axs = plt.subplots(2, sharex=True, figsize=(20, 10))
    axs[0].plot(clusters, variances)    
    axs[0].set_title("Variances")
    axs[0].set_xlabel("Clusters")
    axs[1].plot(clusters, errors)
    axs[1].set_title("Errors")
    axs[1].set_xlabel("Clusters")
    plt.savefig('plot.png')

    functions.testVect(classes, model)

    X, Y = functions.apprentissage(N)
    functions.KDA(X, Y)

if __name__ == '__main__':
    main()
