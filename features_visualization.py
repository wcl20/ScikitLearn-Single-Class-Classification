import argparse
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to training features")
    parser.add_argument("--test", required=True, help="Path to validation features")
    args = parser.parse_args()

    print("[INFO] Reading data ...")
    train_db = h5py.File(args.train, "r")
    X_train = train_db["data"][:]
    y_train = train_db["labels"][:]
    print(f"[INFO] Training Data: {X_train.shape}. Training Labels: {y_train.shape}")

    test_db = h5py.File(args.test, "r")
    X_test = test_db["data"][:]
    y_test = test_db["labels"][:]
    print(f"[INFO] Testing Data: {X_test.shape}. Testing Labels: {y_test.shape}")

    features = np.r_[X_train, X_test]
    features = TSNE(n_components=2).fit_transform(features)

    # Train/Test split
    train_size = X_train.shape[0]
    train, test = features[:train_size], features[train_size:]

    # Find the split between class 0 and class 1
    train_i = y_train.tolist().index(1)
    train_negative, train_positive = train[:train_i], train[train_i:]

    test_i = y_test.tolist().index(1)
    test_negative, test_positive = test[:test_i], test[test_i:]

    print(f"Train negative: {train_negative.shape}")
    print(f"Train positive: {train_positive.shape}")
    print(f"Test negative: {test_negative.shape}")
    print(f"Test positive: {test_positive.shape}")

    plt.scatter(train_negative[:, 0], train_negative[:, 1], c='tomato', alpha=0.1, label="Not Food (Train)")
    plt.scatter(train_positive[:, 0], train_positive[:, 1], c='deepskyblue', alpha=0.1, label="Food (Train)")
    plt.scatter(test_negative[:, 0], test_negative[:, 1], c='red', alpha=0.1, label="Not Food (Test)")
    plt.scatter(test_positive[:, 0], test_positive[:, 1], c='blue', alpha=0.1, label="Food (Test)")
    plt.legend()
    plt.savefig("output/features.png")

if __name__ == '__main__':
    main()
