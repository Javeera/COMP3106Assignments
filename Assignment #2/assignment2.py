# Fatema Lokhandwala (SN: 101259465)
# Gurleen (SN: 101260100)
# Javeera (SN: 101191910)

#written part
#q1 and q4
#q5 and q6
#q2 and q3 Fatema Lokhandwala

import csv
import math
from collections import defaultdict

# helper functions
def mean(values):
  return sum(values) / len(values)

def standard_deviation(values, eps=1e-6):
  pop_mean = mean(values)
  std_deviation = math.sqrt(sum((x - pop_mean) ** 2 for x in values) / len(values))

  # prevent division by zero in case std_deviation is 0 (if all values in a class feature are identical)
  std_deviation = max(std_deviation, eps)
  return std_deviation 

def feature_probability(x, pop_mean, std_deviation):
  first_term = 1.0 / (math.sqrt(2.0 * math.pi * (std_deviation ** 2)) )
  exp = (x - pop_mean) / std_deviation
  return first_term * math.exp(-0.5 * (exp ** 2))
   
def read_snake_dataset(path):
  rows = []

  with open(path, 'r') as f:
    reader = csv.reader(f)
    for i, cols in enumerate(reader):
      # Skip completely empty lines
      if not cols or all((c.strip() == "" for c in cols)):
        continue

      cls = cols[0].strip().lower()
      try:
          length = float(cols[1])
          weight = float(cols[2])
          speed  = float(cols[3])
      except ValueError as e:
          print(f"[warn] line {i}: numeric parse error ({e}); skipping. Row={cols}")
          continue
        
      rows.append({
        "class": cls,
        "length": length,
        "weight": weight,
        "speed":  speed,
      })
  return rows #returns array of rows with classes, links weights and speed


def naive_bayes_classifier(dataset_filepath, snake_measurements):
  # Load the dataset
  dataset = read_snake_dataset(dataset_filepath)

  # Organize the dataset by class
  class_features = defaultdict(lambda: {"length": [], "weight": [], "speed": []})
  class_counts = defaultdict(int)

  for row in dataset:
    cls = row["class"]
    class_counts[cls] += 1
    class_features[cls]["length"].append(row["length"])
    class_features[cls]["weight"].append(row["weight"])
    class_features[cls]["speed"].append(row["speed"])

  total_snakes = sum(class_counts.values())

  # Calculate the mean, standard deviation, and priors for each class
  class_stats = {}
  for cls in class_counts:
    class_stats[cls] = {
      "mean_length": mean(class_features[cls]["length"]),
      "std_length": standard_deviation(class_features[cls]["length"]),
      "mean_weight": mean(class_features[cls]["weight"]),
      "std_weight": standard_deviation(class_features[cls]["weight"]),
      "mean_speed": mean(class_features[cls]["speed"]),
      "std_speed": standard_deviation(class_features[cls]["speed"]),
      "prior": class_counts[cls] / total_snakes
    }
        
    # dataset_filepath is the full file path to a CSV file containing the dataset
    # snake_measurements is a list of [length, weight, speed] measurements for a snake

    # most_likely_class is a string indicating the most likely class, either "anaconda", "cobra", or "python"
    # class_probabilities is a three element list indicating the probability of each class in the order [anaconda probability, cobra probability, python probability]
  length, weight, speed = snake_measurements
  class_probabilities = []

  for species in ["anaconda", "cobra", "python"]:
        stats = class_stats[species]
        p_length = feature_probability(length, stats["mean_length"], stats["std_length"])
        p_weight = feature_probability(weight, stats["mean_weight"], stats["std_weight"])
        p_speed  = feature_probability(speed,  stats["mean_speed"],  stats["std_speed"])

        #naive bayes: multiply conditional probabilities with prior probability
        posterior = stats["prior"] * p_length * p_weight * p_speed
        class_probabilities.append(posterior)

  #normalize probabilities
  #sum = 1
  total = sum(class_probabilities)
  class_probabilities = [p / total for p in class_probabilities]

  #get the max (most likely class)
  species_list = ["anaconda", "cobra", "python"]
  most_likely_class = species_list[class_probabilities.index(max(class_probabilities))]

  return most_likely_class, class_probabilities


#main for testing
if __name__ == "__main__":
    import csv
    import os

    for i in range(4):
        example_folder = f"Examples/Example{i}"

        # --- Read snake measurements (handles [ ] or no brackets) ---
        with open(f"{example_folder}/snake_measurements.txt", 'r') as f:
            line = f.readline().strip()
            line = line.replace('[', '').replace(']', '')
            snake_measurements = [float(x) for x in line.split(',') if x.strip()]

        # --- Read expected most likely class ---
        with open(f"{example_folder}/most_likely_class.txt", 'r') as f:
            expected_class = f.readline().strip().lower()

        # --- Read expected probabilities ---
        with open(f"{example_folder}/class_probabilities.txt", 'r') as f:
            line = f.readline().strip()
            line = line.replace('[', '').replace(']', '')
            expected_probs = [float(x) for x in line.split(',') if x.strip()]

        # --- Run classifier ---
        cls, probs = naive_bayes_classifier(f"{example_folder}/dataset.csv", snake_measurements)

        # --- Print comparison ---
        print(f"\n===== Example{i} =====")
        print(f"Snake measurements: {snake_measurements}")
        print(f"Predicted class: {cls}")
        print(f"Expected  class: {expected_class}")

        print("\nPredicted probabilities:")
        print([round(p, 6) for p in probs])
        print("Expected probabilities:")
        print([round(p, 6) for p in expected_probs])

        # --- Simple correctness check ---
        correct_class = cls.lower() == expected_class
        prob_diff = sum(abs(a - b) for a, b in zip(probs, expected_probs))
        print("\n✅ Class Match:" if correct_class else "\n❌ Class Mismatch!", expected_class)
        print(f"🔹 Total probability difference: {prob_diff:.6f}")
