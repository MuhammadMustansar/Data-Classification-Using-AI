"""
DecodeLabs | Industrial Training Kit - Batch 2026
Project 2: Data Classification Using AI 🤖
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A foundational supervised learning classifier demonstrating:
  • Data handling & preprocessing
  • Train-test splitting (80-20 split with shuffle)
  • Feature scaling (StandardScaler)
  • K-Nearest Neighbors (KNN) algorithm
  • Model evaluation (Confusion Matrix, F1 Score)
  • Performance metrics & diagnostics

Dataset: Iris Benchmark (150 samples, 3 classes, 4 features)
Author: DecodeLabs Training Program
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')


class IrisClassificationModel:
    """
    Supervised Learning Classification Model using K-Nearest Neighbors.
    
    The Predictive Pipeline:
    1. INPUT: Load & understand dataset (Iris Benchmark)
    2. PROCESS: Split data, scale features, train KNN
    3. OUTPUT: Evaluate with confusion matrix & F1 score
    """
    
    def __init__(self, k_neighbors=5, test_size=0.2, random_state=42):
        """
        Initialize the classification model.
        
        Args:
            k_neighbors (int): Number of neighbors for KNN (default: 5)
            test_size (float): Proportion of data for testing (default: 0.2 = 20%)
            random_state (int): Random seed for reproducibility
        """
        self.k_neighbors = k_neighbors
        self.test_size = test_size
        self.random_state = random_state
        
        # Model components
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=k_neighbors)
        
        # Data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        
        # Performance metrics
        self.metrics = {}
        self.confusion_matrix_result = None
        
        # Dataset info
        self.class_names = None
        self.feature_names = None
        
    # ========== PHASE 1: DATA LOADING & UNDERSTANDING ==========
    def load_data(self):
        """
        Load the Iris dataset and understand its structure.
        
        Dataset Specifications:
        - Samples: 150 (balanced across 3 classes)
        - Classes: 3 (Setosa, Versicolor, Virginica)
        - Features: 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
        
        Returns:
            tuple: (X, y) feature matrix and target vector
        """
        # Load Iris dataset (included with scikit-learn)
        iris = load_iris()
        X = iris.data  # Features: (150, 4)
        y = iris.target  # Labels: (150,) with values [0, 1, 2]
        
        # Store metadata
        self.class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']
        self.feature_names = iris.feature_names
        
        print("\n" + "="*70)
        print("PHASE 1: DATA LOADING & UNDERSTANDING")
        print("="*70)
        print(f"Dataset: Iris Benchmark")
        print(f"Samples: {X.shape[0]}")
        print(f"Classes: {len(self.class_names)} → {list(self.class_names)}")
        print(f"Features: {X.shape[1]} → {self.feature_names}")
        print(f"Shape: X={X.shape}, y={y.shape}")
        
        return X, y
    
    # ========== PHASE 2: PREPROCESSING & SPLITTING ==========
    def preprocess_data(self, X, y):
        """
        Preprocess data: Split and Scale.
        
        Step 1: Shuffle & Split (80-20 ratio)
        - Shuffling removes order bias
        - 80% training (learn patterns)
        - 20% testing (validate)
        
        Step 2: Feature Scaling (The Gatekeeper Rule)
        - Transform all features to mean=0, variance=1
        - Prevents features with larger values from dominating
        - StandardScaler: (X - mean) / std_dev
        """
        print("\n" + "="*70)
        print("PHASE 2: PREPROCESSING & STRUCTURAL INTEGRITY")
        print("="*70)
        
        # Initialize class names if not already done
        if self.class_names is None:
            from sklearn.datasets import load_iris
            iris = load_iris()
            self.class_names = iris.target_names
        
        # Step 1: Train-Test Split with shuffle
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,  # 20% for testing
            random_state=self.random_state,
            shuffle=True,  # Randomize order before splitting
            stratify=y  # Maintain class distribution
        )
        
        print(f"\n✓ STRUCTURAL INTEGRITY: THE SPLIT")
        print(f"  Training set: {self.X_train.shape[0]} samples (80%)")
        print(f"  Testing set:  {self.X_test.shape[0]} samples (20%)")
        print(f"  Class distribution (training):")
        for class_idx, class_name in enumerate(self.class_names):
            count = np.sum(self.y_train == class_idx)
            print(f"    {class_name}: {count} samples")
        
        # Step 2: Feature Scaling (StandardScaler)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\n✓ THE GATEKEEPER RULE: FEATURE SCALING")
        print(f"  Method: StandardScaler (mean=0, variance=1)")
        print(f"  Training data range: [{self.X_train.min():.2f}, {self.X_train.max():.2f}]")
        print(f"  Testing data range:  [{self.X_test.min():.2f}, {self.X_test.max():.2f}]")
        
    # ========== PHASE 3: MODEL TRAINING ==========
    def train(self):
        """
        Train the K-Nearest Neighbors model.
        
        The Proximity Principle:
        KNN finds the K nearest training samples (by Euclidean distance)
        and assigns the test sample to the majority class among them.
        
        Workflow: Instantiate → Fit → Predict
        """
        print("\n" + "="*70)
        print("PHASE 3: MODEL TRAINING (THE ALGORITHM)")
        print("="*70)
        
        print(f"\n✓ THE ALGORITHM: K-NEAREST NEIGHBORS")
        print(f"  K value: {self.k_neighbors}")
        print(f"  Distance metric: Euclidean")
        print(f"  Principle: Similar things exist in close proximity")
        
        # Fit the model to training data
        self.model.fit(self.X_train, self.y_train)
        
        print(f"\n✓ INSTANTIATE → FIT → (PREDICT)")
        print(f"  Model instantiated with K={self.k_neighbors}")
        print(f"  Model fitted to training data")
        print(f"  Training samples memorized: {self.X_train.shape[0]}")
        
    # ========== PHASE 4: PREDICTION & EVALUATION ==========
    def evaluate(self):
        """
        Make predictions and evaluate model performance.
        
        Output Validation:
        - Confusion Matrix: TP, FP, FN, TN
        - F1 Score: Harmonic mean of precision and recall
        - Accuracy: Percentage of correct predictions
        """
        print("\n" + "="*70)
        print("PHASE 4: PREDICTION & EVALUATION")
        print("="*70)
        
        # Make predictions on test set
        self.y_pred = self.model.predict(self.X_test)
        
        # Calculate confusion matrix
        self.confusion_matrix_result = confusion_matrix(self.y_test, self.y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        recall = recall_score(self.y_test, self.y_pred, average='weighted')
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"\n✓ OUTPUT VALIDATION: PERFORMANCE METRICS")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        print(f"\n✓ CONFUSION MATRIX:")
        print(f"     Predicted: Setosa | Versi | Virgin")
        for i, class_name in enumerate(self.class_names):
            row_str = f"  {class_name[:6]}: "
            for j in range(3):
                row_str += f"{self.confusion_matrix_result[i, j]:>6} | "
            print(row_str)
        
        print(f"\n✓ DETAILED CLASSIFICATION REPORT:")
        print(classification_report(self.y_test, self.y_pred, 
                                   target_names=self.class_names))
        
    # ========== PHASE 5: CROSS-VALIDATION ==========
    def cross_validate(self, cv=5):
        """
        Perform K-fold cross-validation for robustness.
        
        This tests model performance on different data splits
        to ensure it generalizes well.
        """
        print("\n" + "="*70)
        print("PHASE 5: CROSS-VALIDATION (ROBUSTNESS CHECK)")
        print("="*70)
        
        # Prepare full dataset for cross-validation
        X_full, y_full = self.load_data()
        X_scaled = self.scaler.fit_transform(X_full)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y_full,
            cv=cv, scoring='f1_weighted'
        )
        
        print(f"\n✓ {cv}-FOLD CROSS-VALIDATION RESULTS:")
        for i, score in enumerate(cv_scores, 1):
            print(f"  Fold {i}: {score:.4f} ({score*100:.2f}%)")
        
        print(f"\n  Mean F1 Score:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Min F1 Score:   {cv_scores.min():.4f}")
        print(f"  Max F1 Score:   {cv_scores.max():.4f}")
        
        return cv_scores
    
    # ========== PHASE 6: PREDICTION ON NEW DATA ==========
    def predict_sample(self, features):
        """
        Make prediction on a new sample.
        
        Args:
            features (array-like): Feature vector [sepal_length, sepal_width, 
                                                    petal_length, petal_width]
        
        Returns:
            tuple: (predicted_class_name, confidence_probabilities)
        """
        # Scale the features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # Get distances to nearest neighbors for confidence
        distances, indices = self.model.kneighbors(features_scaled)
        
        # Calculate confidence (inverse of average distance)
        avg_distance = distances[0].mean()
        confidence = 1.0 / (1.0 + avg_distance)
        
        return self.class_names[prediction], confidence
    
    # ========== FULL PIPELINE ==========
    def run_full_pipeline(self):
        """Execute the complete classification pipeline."""
        print("\n" + "="*70)
        print("🤖 DECODELABS PROJECT 2: DATA CLASSIFICATION PIPELINE")
        print("="*70)
        
        # Load data
        X, y = self.load_data()
        
        # Preprocess
        self.preprocess_data(X, y)
        
        # Train
        self.train()
        
        # Evaluate
        self.evaluate()
        
        # Cross-validate
        self.cross_validate(cv=5)
        
        # Summary
        self._print_summary()
    
    def _print_summary(self):
        """Print final session summary."""
        print("\n" + "="*70)
        print("📊 FINAL SUMMARY")
        print("="*70)
        print(f"\nModel: K-Nearest Neighbors (K={self.k_neighbors})")
        print(f"Dataset: Iris (150 samples, 3 classes, 4 features)")
        print(f"Train-Test Split: 80-20 with shuffle")
        print(f"Feature Scaling: StandardScaler (mean=0, var=1)")
        print(f"\nKey Performance Metrics:")
        print(f"  • Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"  • Precision: {self.metrics['precision']:.4f}")
        print(f"  • Recall:    {self.metrics['recall']:.4f}")
        print(f"  • F1 Score:  {self.metrics['f1_score']:.4f}")
        print("\n✓ Project 2 Specification: COMPLETE")
        print("="*70 + "\n")


# ========== EXAMPLE PREDICTIONS ==========
def example_predictions(model):
    """Demonstrate predictions on sample data."""
    print("\n" + "="*70)
    print("🔮 EXAMPLE PREDICTIONS ON NEW DATA")
    print("="*70)
    
    # Sample iris flowers
    samples = [
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.2, 2.9, 4.3, 1.3],  # Versicolor
        [7.1, 3.0, 5.9, 2.1],  # Virginica
    ]
    
    for features in samples:
        predicted_class, confidence = model.predict_sample(features)
        print(f"\nSample: {features}")
        print(f"  Predicted: {predicted_class.upper()} (confidence: {confidence:.2%})")


# ========== ENTRY POINT ==========
if __name__ == "__main__":
    # Create model
    classifier = IrisClassificationModel(k_neighbors=5, test_size=0.2)
    
    # Run full pipeline
    classifier.run_full_pipeline()
    
    # Example predictions
    example_predictions(classifier)
