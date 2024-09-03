import nltk
from nltk.classify import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from nltk.corpus import senseval
from nltk.classify.util import accuracy

instances = senseval.instances('serve.pos')

# Feature extraction function
def senseval_features(instance):
    features = {}
    for word in instance.context:
        features[f'{word[0]}'] = True
    return features
feature_sets = [(senseval_features(instance), instance.senses[0]) for instance in instances]

train_set, test_set = train_test_split(feature_sets, test_size=0.1, random_state=42, shuffle=True)
classifier = NaiveBayesClassifier.train(train_set)

predictions = []
correct_labels = []
comparison = []

for features, label in test_set:
    prediction = classifier.classify(features)
    predictions.append(prediction)
    correct_labels.append(label)
    comparison.append(prediction == label)


acc = accuracy(classifier, test_set)
with open('output.txt', 'w') as output_file:
    for prediction, correct_label, comp in zip(predictions, correct_labels, comparison):
        output_file.write(f"Prediction: {prediction}, Correct Label: {correct_label}, {'True' if comp else 'False'}\n")
    output_file.write(f"\nAccuracy: {acc * 100:.2f}%\n")

# Print the accuracy
print(f"Accuracy: {acc * 100:.2f}%")