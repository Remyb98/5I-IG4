import functions
import matplotlib.pyplot as plt

N = 7


def find_N(classes):
    clusters = [2 ** x for x in range(2, N + 1)]
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


def main():
    classes = [
        './classes/dolphin',
        './classes/flamingo',
        './classes/panda',
        './classes/wild_cat'
    ]

    cluster = N ** 2
    variance, error, model = functions.vocabulaire(cluster, classes[1:3])

    functions.test_vect(classes[1:3], model)

    X, Y = functions.apprentissage(cluster)
    # functions.KDA(X, Y)

    functions.learn_svc(X, Y, model)


if __name__ == '__main__':
    main()
