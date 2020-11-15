import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


def main():
    # Generate training data
    X_train = np.random.randn(100, 2) * 0.3
    # Generate testing data
    X_test = np.random.randn(20, 2) * 0.3

    # Skewed cluster
    # mean = [0, 0]
    # covariance = [[0.4, 0], [0, 0.1]]
    # # Generate training data
    # X_train = np.random.multivariate_normal(mean, covariance, 100)
    # # Generate testing data
    # X_test = np.random.multivariate_normal(mean, covariance, 20)

    # Two clusters
    # # Generate training data
    # X_train = np.random.randn(100, 2) * 0.3
    # X_train = np.r_[X_train + 2, X_train - 2]
    # # Generate testing data
    # X_test = np.random.randn(20, 2) * 0.3
    # X_test = np.r_[X_test + 2, X_test - 2]

    # Generate outliers
    X_outliers = np.random.uniform(low=-4, high=4, size=(40, 2))

    # Train model using only positives
    model = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=0.1)
    model.fit(X_train)

    # Plot graph
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    # Plot model
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    # Plot training data
    plt.scatter(X_train[:, 0], X_train[:, 1], c='white', edgecolors='k', s=40, label='training')
    # Plot testing data
    plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', edgecolors='k', s=40, label='testing')
    # Plot outliers data
    plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', edgecolors='k', s=40, label='outliers')
    plt.legend()
    plt.savefig("output/figure.png")







if __name__ == '__main__':
    main()
