import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset as proxy for aircraft engine sensor data
print("Loading aircraft engine sensor data (Iris dataset as proxy)...")
data = load_iris()
X = data.data  # Sensor readings: sepal length, sepal width, petal length, petal width
y = data.target  # Engine health states: 0=Normal, 1=Warning, 2=Critical

print(f"Dataset shape: {X.shape}")
print(f"Target classes: {data.target_names}")
print(f"Feature names: {data.feature_names}")
print(f"Class distribution: {np.bincount(y)}")
print()

# Split the data into training and test sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Data split completed:")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print()

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
}

# Evaluate each model
for model_name, model in models.items():
    print(f"=== {model_name} Evaluation ===")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Perform 5-Fold Cross-Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    
    print("\n5-Fold Cross-Validation Results:")
    print("Individual fold accuracies:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Analyze overfitting/underfitting
    print("\nPerformance Analysis:")
    accuracy_gap = train_accuracy - test_accuracy
    
    if accuracy_gap > 0.05:
        print("POTENTIAL OVERFITTING: Large gap between training and test accuracy")
        print(f"   Accuracy gap: {accuracy_gap:.4f}")
    elif test_accuracy < 0.7:  # Adjust threshold based on your requirements
        print("POTENTIAL UNDERFITTING: Low performance on both training and test sets")
    else:
        print(" GOOD GENERALIZATION: Model performs well on both training and test sets")
    
    print(f"   Training vs Test difference: {accuracy_gap:.4f}")
    
    # Compare with cross-validation
    if abs(test_accuracy - cv_scores.mean()) > 0.05:
        print("   Note: Test accuracy differs significantly from mean CV accuracy")
    
    print("-" * 50)
    print()

# Additional analysis focusing on safety-critical aspects
print("=== SAFETY-CRITICAL ANALYSIS ===")
print("In aviation applications, consider:")
print("1. False negatives (missing critical warnings) are more dangerous than false positives")
print("2. Model consistency across different data splits is crucial")
print("3. Cross-validation provides better estimate of real-world performance")
print("4. Regular monitoring and retraining may be necessary as engine conditions change")

# Feature importance analysis (for Random Forest)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

print("\nFeature Importance (Random Forest):")
feature_importance = pd.DataFrame({
    'feature': data.feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)