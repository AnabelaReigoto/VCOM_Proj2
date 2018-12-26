import os
import numpy as np
import matplotlib
#matplotlib.use('PS')
import matplotlib.pyplot as plt
from scipy.io import loadmat
#from sklearn.utils import shuffle
#from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression

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

    #create KNN classifier
    knn_classifier = KNeighborsClassifier(weights="uniform",n_neighbors=3)
    
    #train model 
    knn_classifier.fit(X_train,y_train)

    #test
    some_digit = X_test[5000] # taking the ... th image
    pred = knn_classifier.predict([some_digit])
    
    first_image = some_digit
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    #plt.show()
    print (pred)

    #cross-validation to train the model
    #score = cross_val_score(knn_classifier,X_train, y_train,cv=3, scoring="accuracy")
    score = knn_classifier.score(X_test,y_test)

    print(score)

    # standardize the input features and re-train model
    """scaler = StandardScaler()    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # get the predictions that are used to create the confusion matrix
    y_train_pred = cross_val_predict(knn_classifier,X_train,y_train, cv = 3)    
    # plot the confusion matrix
    plot_confusion_matrix(confusion_matrix(y_train,y_train_pred))
    """

    

    #print("TRAINING IMAGES:", len(X_train))
    #print("TEST IMAGES:", len(X_test))


if __name__ == '__main__':
    main()