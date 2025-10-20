import csv
import math
from collections import defaultdict

#helper functions
def mean(values):
    return sum(values) / len(values)

def standard_deviation(values, eps=1e-6):
  pop_mean = mean(values)
  std_deviation = math.sqrt(sum((x - pop_mean) ** 2 for x in values) / len(values))

  # prevent division by zero in case std_deviation is 0 (if all values in a class feature are identical)
  std_deviation = max(std_deviation, eps)

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

      return rows


        




def naive_bayes_classifier(dataset_filepath, snake_measurements):
  # dataset_filepath is the full file path to a CSV file containing the dataset
  # snake_measurements is a list of [length, weight, speed] measurements for a snake

  # most_likely_class is a string indicating the most likely class, either "anaconda", "cobra", or "python"
  # class_probabilities is a three element list indicating the probability of each class in the order [anaconda probability, cobra probability, python probability]
  return most_likely_class, class_probabilities