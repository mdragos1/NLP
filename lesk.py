import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk as simplified_lesk
from nltk.tokenize import word_tokenize
from collections import Counter

# Ensure the WordNet corpus is available
nltk.download('wordnet')
nltk.download('punkt')

# Function to compute the overlap score between two glosses
def gloss_overlap_score(gloss1, gloss2):
    gloss1_tokens = set(word_tokenize(gloss1))
    gloss2_tokens = set(word_tokenize(gloss2))
    overlap = gloss1_tokens.intersection(gloss2_tokens)
    return len(overlap)

# Original Lesk algorithm
def original_lesk(context_sentence, ambiguous_word):
    best_sense = None
    max_overlap = 0
    context = set(word_tokenize(context_sentence))
    for synset in wn.synsets(ambiguous_word):
        gloss = synset.definition()
        examples = synset.examples()
        gloss += ' ' + ' '.join(examples)
        gloss_tokens = set(word_tokenize(gloss))
        overlap = len(context.intersection(gloss_tokens))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = synset
    return best_sense

# Example usage
sentence = "I went to the bank to deposit some money."
ambiguous_word = "bank"

# Apply Original Lesk algorithm
lesk_sense = original_lesk(sentence, ambiguous_word)
print(f"Original Lesk Algorithm: {lesk_sense.definition()}")

# Apply Simplified Lesk algorithm from NLTK
simplified_lesk_sense = simplified_lesk(word_tokenize(sentence), ambiguous_word)
print(f"Simplified Lesk Algorithm: {simplified_lesk_sense.definition()}")
