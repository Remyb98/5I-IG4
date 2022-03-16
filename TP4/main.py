import functions
import matplotlib.pyplot as plt


def main():
    classes = [
        './classes/dolphin',
        './classes/flamingo',
        './classes/panda',
        './classes/wild_cat'
    ]

    # clusters = [2 ** x for x in range(1, 15)]
    clusters = [x for x in range(2, 50)]
    print(clusters)

    variances = list()
    norms = list()

    for cluster in clusters:
        print(f'start for N={cluster}')
        variance, norm = functions.vocabulaire(cluster, classes)
        variances.append(variance)
        norms.append(norm)
    
    fig, axs = plt.subplots(2, sharex=True, figsize=(20, 10))
    axs[0].plot(clusters, variances)
    axs[0].set_title("Variances")
    axs[0].set_xlabel("Clusters")
    axs[1].plot(clusters, norms)
    axs[1].set_title("Norms")
    axs[1].set_xlabel("Clusters")
    plt.savefig('plot.png')


if __name__ == '__main__':
    main()
