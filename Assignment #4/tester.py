from assignment4 import bag_of_words_model
import os

# Helper functions

def load_weights(path):
    with open(path, "r") as f:
        return [float(x) for x in f.read().strip().split(",")]

def load_expected_float_list(path):
    with open(path, "r") as f:
        text = f.read().strip()

    # Normalize separators
    text = text.replace("\n", ",")
    parts = text.split(",")

    # Convert only valid numeric strings → float
    values = []
    for p in parts:
        p = p.strip()
        if p != "":        # ignore empty entries
            values.append(float(p))
    return values

def run_example(example_name):
    print(f"\n\n===== TESTING {example_name} =====")

    base = example_name

    train_dir = os.path.join(base, "training_documents")
    test_doc = os.path.join(base, "test_document.txt")

    # Weight paths
    business_w_path = os.path.join(base, "business_weights.txt")
    entertainment_w_path = os.path.join(base, "entertainment_weights.txt")
    politics_w_path = os.path.join(base, "politics_weights.txt")

    # Expected outputs
    expected_pred_path = os.path.join(base, "prediction.txt")
    expected_scores_path = os.path.join(base, "scores.txt")
    expected_tfidf_path = os.path.join(base, "tf_idf.txt")

    # Load objects
    business_w = load_weights(business_w_path)
    entertainment_w = load_weights(entertainment_w_path)
    politics_w = load_weights(politics_w_path)

    with open(expected_pred_path, "r") as f:
        expected_label = f.read().strip()

    expected_scores = load_expected_float_list(expected_scores_path)
    expected_tfidf = load_expected_float_list(expected_tfidf_path)

    # Build model
    model = bag_of_words_model(train_dir)

    # Compute outputs
    my_tfidf = model.tf_idf(test_doc)
    my_label, my_scores = model.predict(
        test_doc,
        business_w,
        entertainment_w,
        politics_w
    )

    # TF-IDF check
    print("\n--- TF-IDF CHECK ---")
    print("Expected length:", len(expected_tfidf))
    print("Your length:", len(my_tfidf))

    len_match = (len(expected_tfidf) == len(my_tfidf))
    print("Length Match?", "YES" if len_match else "NO")

    tol = 1e-4
    if len_match:
        tfidf_match = all(abs(my_tfidf[i] - expected_tfidf[i]) < tol for i in range(len(my_tfidf)))
    else:
        tfidf_match = False

    print("TF-IDF Values Match?", "YES" if tfidf_match else "NO")

    # Prediction check
    print("\n--- PREDICTION CHECK ---")
    print("Your Prediction:", my_label)
    print("Expected Prediction:", expected_label)
    pred_match = (my_label == expected_label)
    print("Prediction Match?", "YES" if pred_match else "NO")

    # Softmax score check
    print("\n--- SCORES CHECK ---")
    print("Your Scores:", my_scores)
    print("Expected Scores:", expected_scores)

    if len(my_scores) == len(expected_scores):
        scores_match = all(abs(my_scores[i] - expected_scores[i]) < tol for i in range(3))
    else:
        scores_match = False

    print("Scores Match?", "YES" if scores_match else "NO")

    # Final result
    all_pass = len_match and tfidf_match and pred_match and scores_match
    print(f"\nRESULT for {example_name}: ", "PASS 🎉" if all_pass else "FAIL ❌")

# Run all examples
run_example("Examples\Examples\Example0")
run_example("Examples\Examples\Example1")
run_example("Examples\Examples\Example2")
