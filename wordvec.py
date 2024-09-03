import gensim, os, wikipedia
from gensim.models import Word2Vec
import re
from nltk.tokenize import word_tokenize
import nltk
from itertools import combinations
import numpy as np
from tqdm import tqdm

nltk.download('punkt')

modelPath = f"{os.getcwd()}\\GoogleNews-vectors-negative300.bin.gz"
model = gensim.models.KeyedVectors.load_word2vec_format(modelPath, binary=True)


#1
vocab_size = len(model.key_to_index)
print("Number of words in the model's vocabulary:", vocab_size)


#2
page = wikipedia.page(title="Naruto Uzumaki")
article = page.content

tokens = word_tokenize(article)
tokens = [re.sub(r'\W+', '', token) for token in tokens if token.isalpha()]

words_not_in_vocab = [word for word in tokens if word.lower() not in model.key_to_index]
print("Words not in the model's vocabulary:", words_not_in_vocab)


#3
word_list = [word for word in tqdm(tokens) if word.lower() in model.key_to_index]

word_combinations = list(combinations(word_list, 2))
distances = [(word1, word2, abs(model.similarity(word1, word2))) for word1, word2 in tqdm(word_combinations)]

most_distant = min(distances, key=lambda x: x[2])
closest = max(distances, key=lambda x: x[2])

print("Most distant words:", most_distant)
print("Closest words:", closest)


#4
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp(article)
named_entities = [ent.text for ent in doc.ents]

print("Named Entities:", named_entities)

similar_words = {}
for entity in tqdm(named_entities):
    entity_lower = entity.lower()
    if entity_lower in model.key_to_index:
        similar_words[entity] = model.most_similar(entity_lower, topn=5)

print("Similar words to named entities:")
for entity, similar in tqdm(similar_words.items()):
    print(f"{entity}: {similar}")


#5
from sklearn.cluster import KMeans

word_vectors = [model[word.lower()] for word in tqdm(tokens) if word.lower() in model.key_to_index]
kmeans = KMeans(n_clusters=5, random_state=0).fit(word_vectors)

clusters = {}
for word, label in tqdm(zip(tokens, kmeans.labels_)):
    if word.lower() in model.key_to_index:
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(word)

print("Clusters of similar words:")
for label, cluster in clusters.items():
    print(f"Cluster {label}: {cluster}")