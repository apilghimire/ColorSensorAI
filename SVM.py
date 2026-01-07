import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


def rgb_to_cielab(r, g, b):
    """Convert RGB to CIELAB color space"""
    # Normalize RGB values
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # Convert to linear RGB
    def gamma_correction(channel):
        if channel > 0.04045:
            return ((channel + 0.055) / 1.055) ** 2.4
        else:
            return channel / 12.92

    r, g, b = gamma_correction(r), gamma_correction(g), gamma_correction(b)

    # Convert to XYZ
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Reference white D65
    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

    # XYZ to Lab
    def f(t):
        if t > 0.008856:
            return t ** (1 / 3)
        else:
            return (7.787 * t) + (16 / 116)

    l = 116 * f(y) - 16
    a = 500 * (f(x) - f(y))
    b_lab = 200 * (f(y) - f(z))

    return l, a, b_lab


def train_svm_classifier():
    print("=" * 60)
    print("SVM COLOR CLASSIFIER WITH CIELAB CONVERSION")
    print("=" * 60)

    # Load data
    df = pd.read_csv('data.csv')
    print(f"\nLoaded {len(df)} samples from data.csv")
    print(f"Labels: {df['Label'].unique()}")

    # Get max values for normalization
    max_red = df['Red'].max()
    max_green = df['Green'].max()
    max_blue = df['Blue'].max()

    # Process each sample
    features = []
    labels = []

    print("\nProcessing samples...")
    for idx, row in df.iterrows():
        # Normalize RGB values
        r_norm = (row['Red'] / max_red) * 255 if row['Red'] > 0 else 0
        g_norm = (row['Green'] / max_green) * 255 if row['Green'] > 0 else 0
        b_norm = (row['Blue'] / max_blue) * 255 if row['Blue'] > 0 else 0

        # Convert to CIELAB
        l, a, b = rgb_to_cielab(r_norm, g_norm, b_norm)

        # Compute intensity
        intensity = (row['Red'] + row['Green'] + row['Blue']) / 3

        # Create feature vector [L, a, b, intensity, distance]
        feature_vector = [l, a, b, intensity, row['Distance_mm']]
        features.append(feature_vector)
        labels.append(row['Label'])

    X = np.array(features)
    y = np.array(labels)

    print(f"Feature shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM classifier with hyperparameter tuning
    print("\nTraining SVM classifier with hyperparameter tuning...")
    print("(This may take a few moments...)")

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Best model
    best_svm = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")

    # Predictions
    y_pred = best_svm.predict(X_test_scaled)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"SVM CLASSIFIER ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('SVM Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('svm_confusion_matrix.png')
    print("\nConfusion matrix saved as 'svm_confusion_matrix.png'")

    # Save model and preprocessing parameters
    joblib.dump(best_svm, 'svm_model.pkl')
    joblib.dump(scaler, 'svm_scaler.pkl')

    # Save normalization parameters
    norm_params = {
        'max_red': max_red,
        'max_green': max_green,
        'max_blue': max_blue
    }
    joblib.dump(norm_params, 'svm_norm_params.pkl')

    print("\nModel saved as: svm_model.pkl")
    print("Scaler saved as: svm_scaler.pkl")
    print("Normalization parameters saved as: svm_norm_params.pkl")

    return accuracy


if __name__ == "__main__":
    train_svm_classifier()