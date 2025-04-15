import cv2
import numpy as np
import pandas as pd
from skimage import measure, filters, morphology
from skimage.measure import regionprops_table
from skimage.morphology import convex_hull_image
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import variation

# === Train classifier from WBCD ===
def train_classifier():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_scaled, y_train)

    return clf, scaler

# === Extract 30 features from image ===
def extract_features_from_image(image_path, show_image=False):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = filters.threshold_otsu(blurred)
    binary = blurred > threshold
    cleaned = morphology.remove_small_objects(binary, min_size=50)
    label_image = measure.label(cleaned)

    props = regionprops_table(label_image, intensity_image=gray, properties=[
        'area', 'perimeter', 'equivalent_diameter', 'eccentricity',
        'solidity', 'mean_intensity', 'coords'
    ])
    df = pd.DataFrame(props)

    if df.empty:
        raise ValueError("❌ No nuclei detected in the image.")

    df['radius'] = df['equivalent_diameter'] / 2
    df['smoothness'] = 1 - df['solidity']
    df['compactness'] = df['perimeter'] ** 2 / (4 * np.pi * df['area'])
    df['concavity'] = 1 - df['solidity']
    df['symmetry'] = df['eccentricity']
    df['fractal_dimension'] = df['perimeter'] / df['area']

    concave_points_list = []
    for i in range(len(df)):
        coords = props['coords'][i]
        blank = np.zeros_like(gray)
        for y, x in coords:
            blank[y, x] = 255
        chull = convex_hull_image(blank.astype(bool))
        hull_area = np.sum(chull)
        actual_area = df.loc[i, 'area']
        concave_points = max((hull_area - actual_area) // 10, 1)
        concave_points_list.append(concave_points)
    df['concave_points'] = concave_points_list

    df['texture'] = df['mean_intensity']
    df['texture_var'] = variation(df['mean_intensity'])

    # Make 30 features
    feature_list = []
    features = [
        'radius', 'texture', 'perimeter', 'area', 'smoothness',
        'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'
    ]
    for feat in features:
        values = df[feat] if feat in df else df[feat + '_var']
        feature_list.extend([
            np.mean(values),
            np.std(values) / np.sqrt(len(values)),
            np.max(values)
        ])

    return feature_list

# === Final predictor ===
def predict(features=None, image_path=None):
    clf, scaler = train_classifier()

    if image_path:
        features = extract_features_from_image(image_path)
    elif features:
        if len(features) != 30:
            raise ValueError("❌ You must provide exactly 30 features.")
    else:
        raise ValueError("❌ Provide either an image_path or 30 numeric features.")

    features_scaled = scaler.transform([features])
    prediction = clf.predict(features_scaled)[0]
    label = "Benign (No Cancer)" if prediction == 1 else "Malignant (Cancer)"
    return label
