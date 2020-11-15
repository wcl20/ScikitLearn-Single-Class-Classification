import argparse
import h5py
import joblib
import numpy as np
from sklearn.metrics import classification_report

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--db", required=True, help="Path to HDF5 file")
    parser.add_argument("-m", "--model", required=True, help="Path to model")
    args = parser.parse_args()

    db = h5py.File(args.db, "r")
    data = db["data"][:]
    labels = db["labels"][:]
    print(f"[INFO] Data: {data.shape}. Labels: {labels.shape}")

    # Load model
    model = joblib.load(args.model)

    # Evaluate model
    preds = model.predict(data)
    print(classification_report(labels, np.clip(preds, 0, 1), target_names=db["class_names"][:]))


if __name__ == '__main__':
    main()
