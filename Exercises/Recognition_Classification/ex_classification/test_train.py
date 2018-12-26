import os
import numpy as np
import matplotlib
#matplotlib.use('PS')
import matplotlib.pyplot as plt
from scipy.io import loadmat
#from sklearn.utils import shuffle
#from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

def save_fig(fig_id, tight_layout=True):
    path = "images"
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_confusion_matrix(matrix):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    save_fig("confusion_matrix_plot", tight_layout=False)

def main():
    ## load MNIST data
    mnist_raw = loadmat('mnist-original.mat')
    X, y = mnist_raw["data"].T, mnist_raw["label"][0]

    # separate dataset: first 60000 train set, remaining 10000 test set
    X_train, y_train = X[:60000:], y[:60000] 
    X_test, y_test = X[60000:], y[60000:]

    # shuffle train set
    np.random.seed(42)
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index],y_train[shuffle_index]

       
    for i in range(0,6):
        first_image = X_train[i]
        first_image = np.array(first_image, dtype='float')
        pixels = first_image.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()


    
    #print("TRAINING IMAGES:", len(X_train))
    #print("TEST IMAGES:", len(X_test))


if __name__ == '__main__':
    main()