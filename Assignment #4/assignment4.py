# Name this file assignment4.py when you submit

import os
import math

class bag_of_words_model:

  def __init__(self, directory):

    self.documents = []

    # Sort file names to guarantee professor’s order
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"):
            full_path = os.path.join(directory, filename)
            with open(full_path, "r") as f:
                words = f.read().strip().split()
                self.documents.append(words)

    # Build vocabulary (alphabetical)
    vocab_set = set()
    for doc in self.documents:
        vocab_set.update(doc)

    self.vocab = sorted(list(vocab_set))

    # Compute IDF with log base 2
    N = len(self.documents)
    self.idf_vector = []
    for word in self.vocab:
        df = sum(1 for doc in self.documents if word in doc)
        idf = math.log2(N / df)
        self.idf_vector.append(idf)

    self.word_to_index = {word: i for i, word in enumerate(self.vocab)}

    # Return nothing


  def tf_idf(self, document_filepath):
    # document_filepath is the full file path to a test document

    with open(document_filepath, "r") as f:
      words = f.read().strip().split()

    # Term frequency vector
    tf_vector = [0] * len(self.vocab)
    total_words = len(words)

    for w in words:
      if w in self.word_to_index:
        idx = self.word_to_index[w]
        tf_vector[idx] += 1

    if total_words > 0:
      tf_vector = [count / total_words for count in tf_vector]

    # Now multiply by IDF
    tf_idf_vector = [tf_vector[i] * self.idf_vector[i] for i in range(len(self.vocab))]

    # # Compute TF-IDF
    # tf_idf_vector = []
    # for i in range(len(self.vocab)):
    #   tfidf = tf_vector[i] * self.idf_vector[i]
    #   tf_idf_vector.append(tfidf)

    # Return the term frequency-inverse document frequency vector for the document
    return tf_idf_vector


  def predict(self, document_filepath, business_weights, entertainment_weights, politics_weights):
    # document_filepath is the full file path to a test document
    # business_weights is a list of weights for the business artificial neuron
    # entertainment_weights is a list of weights for the entertainment artificial neuron
    # politics_weights is a list of weights for the politics artificial neuron

    x = self.tf_idf(document_filepath)

    # Compute raw scores y = w · x
    def dot(w, x):
      return sum(w[i] * x[i] for i in range(len(x)))

    y_business = dot(business_weights, x)
    y_entertainment = dot(entertainment_weights, x)
    y_politics = dot(politics_weights, x)

    # Softmax activation
    exps = [
      math.exp(y_business),
      math.exp(y_entertainment),
      math.exp(y_politics)
    ]
    total = sum(exps)
    scores = [e / total for e in exps]

    # Determine predicted label
    labels = ["business", "entertainment", "politics"]
    predicted_label = labels[scores.index(max(scores))]

    # Return the predicted label and the softmax scores
    return predicted_label, scores
