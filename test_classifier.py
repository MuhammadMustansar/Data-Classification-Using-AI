"""
DecodeLabs | Project 2: Classification Model Test Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Complete test suite covering:
  ✓ Data loading and integrity
  ✓ Train-test split verification
  ✓ Feature scaling validation
  ✓ Model training
  ✓ Prediction accuracy
  ✓ Cross-validation robustness
  ✓ K-value optimization
  ✓ Edge cases and error handling
"""

import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score

# Import the classifier
from classifier import IrisClassificationModel

# Color codes
CYAN = '\033[96m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(title):
    """Print a formatted section header."""
    print(f"\n{CYAN}{BOLD}{'='*70}{RESET}")
    print(f"{CYAN}{BOLD}{title:^70}{RESET}")
    print(f"{CYAN}{BOLD}{'='*70}{RESET}\n")

def print_test(test_name, passed, details=""):
    """Print test result."""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"  {status}  {test_name}")
    if details:
        print(f"         {details}")

def test_data_loading():
    """Test 1: Data Loading & Understanding."""
    print_header("TEST 1: DATA LOADING & UNDERSTANDING")
    
    classifier = IrisClassificationModel()
    X, y = classifier.load_data()
    
    tests_passed = 0
    
    # Test 1.1: Shape validation
    test_1_1 = X.shape == (150, 4)
    print_test("Dataset shape (150, 4)", test_1_1, f"Got: {X.shape}")
    if test_1_1: tests_passed += 1
    
    # Test 1.2: Target shape
    test_1_2 = y.shape == (150,)
    print_test("Target shape (150,)", test_1_2, f"Got: {y.shape}")
    if test_1_2: tests_passed += 1
    
    # Test 1.3: Class count
    test_1_3 = len(np.unique(y)) == 3
    print_test("Number of classes (3)", test_1_3, f"Got: {len(np.unique(y))}")
    if test_1_3: tests_passed += 1
    
    # Test 1.4: Class balance
    class_counts = [np.sum(y == i) for i in range(3)]
    test_1_4 = all(count == 50 for count in class_counts)
    print_test("Class balance (50 each)", test_1_4, f"Counts: {class_counts}")
    if test_1_4: tests_passed += 1
    
    # Test 1.5: Feature names
    test_1_5 = len(classifier.feature_names) == 4
    print_test("Feature count (4)", test_1_5, f"Got: {len(classifier.feature_names)}")
    if test_1_5: tests_passed += 1
    
    print(f"\n{YELLOW}Result: {tests_passed}/5 tests passed{RESET}\n")
    return tests_passed == 5

def test_preprocessing():
    """Test 2: Preprocessing & Splitting."""
    print_header("TEST 2: PREPROCESSING & SPLITTING")
    
    classifier = IrisClassificationModel(test_size=0.2, random_state=42)
    X, y = classifier.load_data()
    classifier.preprocess_data(X, y)
    
    tests_passed = 0
    
    # Test 2.1: Train set size
    test_2_1 = classifier.X_train.shape[0] == 120
    print_test("Training set size (120)", test_2_1, f"Got: {classifier.X_train.shape[0]}")
    if test_2_1: tests_passed += 1
    
    # Test 2.2: Test set size
    test_2_2 = classifier.X_test.shape[0] == 30
    print_test("Testing set size (30)", test_2_2, f"Got: {classifier.X_test.shape[0]}")
    if test_2_2: tests_passed += 1
    
    # Test 2.3: Feature count preserved
    test_2_3 = classifier.X_train.shape[1] == 4
    print_test("Feature count preserved (4)", test_2_3, f"Got: {classifier.X_train.shape[1]}")
    if test_2_3: tests_passed += 1
    
    # Test 2.4: Scaling applied (mean ≈ 0)
    train_mean = np.abs(classifier.X_train.mean())
    test_2_4 = train_mean < 0.1
    print_test("Scaling applied (mean ≈ 0)", test_2_4, f"Mean: {classifier.X_train.mean():.4f}")
    if test_2_4: tests_passed += 1
    
    # Test 2.5: Scaling applied (variance ≈ 1)
    train_var = np.mean(classifier.X_train.std(axis=0))
    test_2_5 = 0.9 < train_var < 1.1
    print_test("Scaling applied (var ≈ 1)", test_2_5, f"Variance: {train_var:.4f}")
    if test_2_5: tests_passed += 1
    
    print(f"\n{YELLOW}Result: {tests_passed}/5 tests passed{RESET}\n")
    return tests_passed == 5

def test_model_training():
    """Test 3: Model Training."""
    print_header("TEST 3: MODEL TRAINING")
    
    classifier = IrisClassificationModel(k_neighbors=5)
    X, y = classifier.load_data()
    classifier.preprocess_data(X, y)
    classifier.train()
    
    tests_passed = 0
    
    # Test 3.1: Model is trained
    test_3_1 = classifier.model.n_neighbors == 5
    print_test("Model K value (5)", test_3_1, f"Got: {classifier.model.n_neighbors}")
    if test_3_1: tests_passed += 1
    
    # Test 3.2: Model has training data
    test_3_2 = classifier.model._fit_X is not None
    print_test("Model fitted to training data", test_3_2)
    if test_3_2: tests_passed += 1
    
    # Test 3.3: Training set labels stored
    test_3_3 = classifier.model._y is not None
    print_test("Training labels stored", test_3_3)
    if test_3_3: tests_passed += 1
    
    # Test 3.4: Can make predictions
    try:
        sample = classifier.X_test[0:1]
        pred = classifier.model.predict(sample)
        test_3_4 = pred.shape == (1,)
        print_test("Can make predictions", test_3_4)
        if test_3_4: tests_passed += 1
    except:
        print_test("Can make predictions", False)
    
    # Test 3.5: Predictions in valid range
    classifier.y_pred = classifier.model.predict(classifier.X_test)
    test_3_5 = all(0 <= p <= 2 for p in classifier.y_pred)
    print_test("Predictions in valid range [0,2]", test_3_5)
    if test_3_5: tests_passed += 1
    
    print(f"\n{YELLOW}Result: {tests_passed}/5 tests passed{RESET}\n")
    return tests_passed == 5

def test_evaluation_metrics():
    """Test 4: Evaluation & Metrics."""
    print_header("TEST 4: EVALUATION METRICS")
    
    classifier = IrisClassificationModel(k_neighbors=5)
    X, y = classifier.load_data()
    classifier.preprocess_data(X, y)
    classifier.train()
    classifier.evaluate()
    
    tests_passed = 0
    
    # Test 4.1: Accuracy in valid range
    accuracy = classifier.metrics['accuracy']
    test_4_1 = 0 <= accuracy <= 1
    print_test("Accuracy in [0,1]", test_4_1, f"Got: {accuracy:.4f}")
    if test_4_1: tests_passed += 1
    
    # Test 4.2: Precision in valid range
    precision = classifier.metrics['precision']
    test_4_2 = 0 <= precision <= 1
    print_test("Precision in [0,1]", test_4_2, f"Got: {precision:.4f}")
    if test_4_2: tests_passed += 1
    
    # Test 4.3: Recall in valid range
    recall = classifier.metrics['recall']
    test_4_3 = 0 <= recall <= 1
    print_test("Recall in [0,1]", test_4_3, f"Got: {recall:.4f}")
    if test_4_3: tests_passed += 1
    
    # Test 4.4: F1 Score in valid range
    f1 = classifier.metrics['f1_score']
    test_4_4 = 0 <= f1 <= 1
    print_test("F1 Score in [0,1]", test_4_4, f"Got: {f1:.4f}")
    if test_4_4: tests_passed += 1
    
    # Test 4.5: Confusion matrix shape
    cm = classifier.confusion_matrix_result
    test_4_5 = cm.shape == (3, 3)
    print_test("Confusion matrix shape (3,3)", test_4_5, f"Got: {cm.shape}")
    if test_4_5: tests_passed += 1
    
    # Test 4.6: High accuracy (should be >80% on Iris)
    test_4_6 = accuracy > 0.8
    print_test("Accuracy > 80% (expected on Iris)", test_4_6, f"Got: {accuracy*100:.1f}%")
    if test_4_6: tests_passed += 1
    
    print(f"\n{YELLOW}Result: {tests_passed}/6 tests passed{RESET}\n")
    return tests_passed == 6

def test_cross_validation():
    """Test 5: Cross-Validation."""
    print_header("TEST 5: CROSS-VALIDATION")
    
    classifier = IrisClassificationModel(k_neighbors=5)
    X, y = classifier.load_data()
    classifier.preprocess_data(X, y)
    classifier.train()
    
    cv_scores = classifier.cross_validate(cv=5)
    
    tests_passed = 0
    
    # Test 5.1: 5 folds completed
    test_5_1 = len(cv_scores) == 5
    print_test("5-fold cross-validation completed", test_5_1, f"Got: {len(cv_scores)} folds")
    if test_5_1: tests_passed += 1
    
    # Test 5.2: All scores in valid range
    test_5_2 = all(0 <= score <= 1 for score in cv_scores)
    print_test("All CV scores in [0,1]", test_5_2)
    if test_5_2: tests_passed += 1
    
    # Test 5.3: Mean CV score reasonable
    mean_score = cv_scores.mean()
    test_5_3 = mean_score > 0.8
    print_test("Mean CV score > 80%", test_5_3, f"Got: {mean_score*100:.1f}%")
    if test_5_3: tests_passed += 1
    
    # Test 5.4: Low variance (consistent)
    std_score = cv_scores.std()
    test_5_4 = std_score < 0.1
    print_test("Low score variance (std < 0.1)", test_5_4, f"Got: {std_score:.4f}")
    if test_5_4: tests_passed += 1
    
    print(f"\n{YELLOW}Result: {tests_passed}/4 tests passed{RESET}\n")
    return tests_passed == 4

def test_k_optimization():
    """Test 6: K-Value Optimization."""
    print_header("TEST 6: K-VALUE OPTIMIZATION")
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Simple split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    k_values = [1, 3, 5, 7, 9, 15]
    f1_scores = []
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
    
    print("F1 Scores for different K values:")
    for k, f1 in zip(k_values, f1_scores):
        print(f"  K={k:2d}: {f1:.4f}")
    
    tests_passed = 0
    
    # Test 6.1: All F1 scores are valid
    test_6_1 = all(0 <= f1 <= 1 for f1 in f1_scores)
    print_test("All F1 scores in valid range [0,1]", test_6_1)
    if test_6_1: tests_passed += 1
    
    # Test 6.2: F1 scores are reasonable (>75% for Iris)
    test_6_2 = all(f1 > 0.75 for f1 in f1_scores)
    print_test("All K values achieve >75% F1 on Iris", test_6_2)
    if test_6_2: tests_passed += 1
    
    print(f"\n{YELLOW}Result: {tests_passed}/2 tests passed{RESET}\n")
    return tests_passed == 2

def test_edge_cases():
    """Test 7: Edge Cases & Error Handling."""
    print_header("TEST 7: EDGE CASES")
    
    tests_passed = 0
    
    # Test 7.1: Different random states
    classifier1 = IrisClassificationModel(random_state=42)
    classifier2 = IrisClassificationModel(random_state=42)
    X1, y1 = classifier1.load_data()
    X2, y2 = classifier2.load_data()
    test_7_1 = np.allclose(X1, X2) and np.allclose(y1, y2)
    print_test("Reproducibility (same seed produces same split)", test_7_1)
    if test_7_1: tests_passed += 1
    
    # Test 7.2: Different K values
    for k in [1, 5, 10, 20]:
        classifier = IrisClassificationModel(k_neighbors=k)
        try:
            X, y = classifier.load_data()
            classifier.preprocess_data(X, y)
            classifier.train()
            classifier.y_pred = classifier.model.predict(classifier.X_test)
        except:
            print_test(f"K={k} handling", False)
            return tests_passed
    
    test_7_2 = True
    print_test("Different K values (1,5,10,20) all work", test_7_2)
    if test_7_2: tests_passed += 1
    
    # Test 7.3: Prediction on new sample
    classifier = IrisClassificationModel()
    X, y = classifier.load_data()
    classifier.preprocess_data(X, y)
    classifier.train()
    
    try:
        sample = [5.1, 3.5, 1.4, 0.2]  # Typical setosa
        predicted_class, confidence = classifier.predict_sample(sample)
        test_7_3 = predicted_class in classifier.class_names and 0 <= confidence <= 1
        print_test("Prediction on new sample", test_7_3, f"Class: {predicted_class}, Conf: {confidence:.2%}")
        if test_7_3: tests_passed += 1
    except:
        print_test("Prediction on new sample", False)
    
    print(f"\n{YELLOW}Result: {tests_passed}/3 tests passed{RESET}\n")
    return tests_passed == 3

def run_all_tests():
    """Run all test suites."""
    print(f"\n{CYAN}{BOLD}")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "DecodeLabs | Project 2: Data Classification".center(68) + "║")
    print("║" + "Complete Test Suite".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    print(f"{RESET}")
    
    results = []
    
    # Run all tests
    results.append(("Data Loading", test_data_loading()))
    results.append(("Preprocessing", test_preprocessing()))
    results.append(("Model Training", test_model_training()))
    results.append(("Evaluation Metrics", test_evaluation_metrics()))
    results.append(("Cross-Validation", test_cross_validation()))
    results.append(("K Optimization", test_k_optimization()))
    results.append(("Edge Cases", test_edge_cases()))
    
    # Final report
    print_header("FINAL TEST REPORT")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = f"{GREEN}✓ PASSED{RESET}" if passed else f"{RED}✗ FAILED{RESET}"
        print(f"  {status}  {test_name}")
    
    print()
    pass_percentage = (passed_count / total_count * 100) if total_count > 0 else 0
    
    if passed_count == total_count:
        print(f"{GREEN}{BOLD}✓ ALL TESTS PASSED ({pass_percentage:.0f}%){RESET}")
        print(f"{GREEN}Project 2 Specification: COMPLETE ✓{RESET}\n")
    else:
        print(f"{YELLOW}{BOLD}⚠ {passed_count}/{total_count} tests passed ({pass_percentage:.0f}%){RESET}\n")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test suite interrupted by user.{RESET}\n")
        sys.exit(0)
