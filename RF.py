import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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


def train_random_forest_classifier():
    print("=" * 60)
    print("RANDOM FOREST COLOR CLASSIFIER WITH CIELAB CONVERSION")
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

    # Scale features (optional for Random Forest, but helps with consistency)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest classifier with hyperparameter tuning
    print("\nTraining Random Forest classifier with hyperparameter tuning...")
    print("(This may take a few moments...)")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Best model
    best_rf = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")

    # Predictions
    y_pred = best_rf.predict(X_test_scaled)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"RANDOM FOREST CLASSIFIER ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_names = ['L (Lightness)', 'a (Green-Red)', 'b (Blue-Yellow)', 'Intensity', 'Distance']
    importances = best_rf.feature_importances_

    print("\nFeature Importances:")
    for name, importance in zip(feature_names, importances):
        print(f"  {name}: {importance:.4f}")

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importances)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Random Forest Feature Importances')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')
    print("\nFeature importance plot saved as 'rf_feature_importance.png'")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Random Forest Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('rf_confusion_matrix.png')
    print("Confusion matrix saved as 'rf_confusion_matrix.png'")

    # Save model and preprocessing parameters
    joblib.dump(best_rf, 'rf_model.pkl')
    joblib.dump(scaler, 'rf_scaler.pkl')

    # Save normalization parameters
    norm_params = {
        'max_red': df['Red'].max(),
        'max_green': df['Green'].max(),
        'max_blue': df['Blue'].max()
    }
    joblib.dump(norm_params, 'rf_norm_params.pkl')

    print("\nModel saved as: rf_model.pkl")
    print("Scaler saved as: rf_scaler.pkl")
    print("Normalization parameters saved as: rf_norm_params.pkl")

    return accuracy


if __name__ == "__main__":
    train_random_forest_classifier()