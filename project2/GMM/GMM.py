import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import scipy.io as sio

# load data
training_set = sio.loadmat('trainingset.mat')
testing_set = sio.loadmat('testingset.mat')
x_train = training_set['trainingset']
x_test = testing_set['testingset']

dims = [80, 200]
for dim in range(3):
    if dim > 0:
        # PCA 40 and 80
        pca = PCA(n_components=dims[dim - 1])
        pca.fit(x_train)
        # project to 2d for visualization
        pca_2 = PCA(n_components=2)
        pca_2.fit(pca.transform(x_train))
        X = pca_2.transform(pca.transform(x_train))

    else:
        # raw image
        pca_2 = PCA(n_components=2)
        pca_2.fit(x_train)
        X = pca_2.transform(x_train)
    # draw GMM results
    titles = ['raw face image', 'PCA=80', 'PCA=200']
    labels = GaussianMixture(n_components=3).fit_predict(X)
    plt.figure(1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
    plt.title(f'GMM visualization for {titles[dim]}')
    plt.show()

    # draw 3 faces assigned to each component
    index = ['1st', '2nd', '3rd']
    for k in range(3):
        selected = np.random.choice(x_train[labels == 0].shape[0], 3, replace=False)
        fig, axs = plt.subplots(1, 3)
        fig.suptitle(f'{titles[dim]} images belong to the {index[k]} Gaussian component')
        for i in range(3):
            axs[i].imshow(np.transpose(x_train[selected[i], :].reshape(32, 32)), 'gray')
            axs[i].axis('off')
        plt.show()


