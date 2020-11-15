import argparse
import h5py
import joblib
import os
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--db", required=True, help="Path to HDF5 file")
    args = parser.parse_args()

    db = h5py.File(args.db, "r")
    data = db["data"][:]
    labels = db["labels"][:]
    print(f"[INFO] Data: {data.shape}. Labels: {labels.shape}")

    # Get all positive class
    i = labels.tolist().index(1)
    X_train = data[i:]

    # Train model
    model = svm.OneClassSVM(gamma=0.1, kernel='rbf', nu=0.04)
    model.fit(X_train)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # Evaluate model
    preds = model.predict(data)
    print(classification_report(labels, np.clip(preds, 0, 1), target_names=db["class_names"][:]))


if __name__ == '__main__':
    main()
