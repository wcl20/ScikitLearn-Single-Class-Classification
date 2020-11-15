import argparse
import glob
import numpy as np
import joblib
import os
import tqdm
from core.io import HDF5Writer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Path to dataset")
    parser.add_argument("--pca", default=None, help="Path to PCA model")
    parser.add_argument("--scaler", default=None, help="Path to Scaler model")
    parser.add_argument("-o", "--output", required=True, help="Output HDF5 file")
    parser.add_argument("-b", "--buffer", type=int, default=1000, help="Buffer size for feature extraction")
    args = parser.parse_args()

    # Get labels
    img_paths = sorted(glob.glob(f"{args.dataset}/*.jpg"))
    labels = [int(os.path.basename(img_path)[0]) for img_path in img_paths]
    labels = np.array(labels)

    batch_size = 32

    # Feature extraction model
    model = ResNet50(weights="imagenet", include_top=False)
    features = []
    for i in tqdm.tqdm(np.arange(0, len(img_paths), batch_size)):
        # Get batch
        batch_img_paths = img_paths[i:i+batch_size]

        # Process each path
        batch_images = []
        for img_path in batch_img_paths:
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            batch_images.append(image)
        batch_images = np.vstack(batch_images)

        # Extract features
        batch_features = model.predict(batch_images, batch_size=batch_size)
        batch_features = batch_features.reshape((batch_features.shape[0], -1))
        features.append(batch_features)

    features = np.vstack(features)
    print(f"[INFO] Extracted features: {features.shape}")

    # Load models
    scaler = joblib.load(args.scaler) if args.scaler else StandardScaler().fit(features)
    features = scaler.transform(features)
    pca = joblib.load(args.pca) if args.pca else PCA(n_components=1024, whiten=True).fit(features)
    features = pca.transform(features)
    print(f"[INFO] PCA features: {features.shape}")
    print(f"[INFO] PCA Explained variance %: {sum(pca.explained_variance_ratio_)}")

    # Save models
    os.makedirs("models", exist_ok=True)
    if not args.scaler:
        print("[INFO] Saving Scaler model ...")
        joblib.dump(scaler, "models/scaler.pkl")
    if not args.pca:
        print("[INFO] Saving PCA model ...")
        joblib.dump(pca, "models/pca.pkl")

    # Write to hdf5
    dims = (len(img_paths), 1024)
    dataset = HDF5Writer(args.output, dims, buffer_size=args.buffer)
    dataset.store_class_names(["Not Food", "Food"])
    for i in tqdm.tqdm(np.arange(0, len(img_paths), batch_size)):
        # Add to dataset
        batch_features = features[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        dataset.add(batch_features, batch_labels)
    dataset.close()


if __name__ == '__main__':
    main()
