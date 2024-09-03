import wikipedia
import wikipedia
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.tag.stanford import StanfordPOSTagger
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter

# Ex1
page = wikipedia.page(title="Naruto Uzumaki")
article = page.content
title = page.title

print(title)

words = word_tokenize(article)[:200]
print(f"First 200 words: {words}")
sentences = sent_tokenize(article)

## Nltk Tagger
N = 20
for i in range(N):
    sentence = sentences[i]
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)
    print("Sentence", i + 1, ":", sentence)
    print("POS tags:", tagged_words)

## Stanford Tagger
path_to_model = "D:\\Master\\Anul1 Sem2\\NLP\\stanford-postagger-full-2020-11-17\\models\\english-bidirectional-distsim.tagger"
path_to_jar = "D:\\Master\\Anul1 Sem2\\NLP\\stanford-postagger-full-2020-11-17\\stanford-postagger.jar"
java_path = "C:\\Program Files\\Java\\jdk-20\\bin\\java.exe"

os.environ['JAVAHOME'] = java_path
tagger=StanfordPOSTagger(path_to_model, path_to_jar)
tagger.java_options='-mx4096m'

for i in range(20):
    sentence = sentences[i]
    words = word_tokenize(sentence)
    tagged_words = tagger.tag(words)
    print("Sentence", i + 1, ":", sentence)
    print("POS tags:", tagged_words)

# =====================================================================================================================================================
# Ex2

def get_words_by_pos_tag(pos_tag, text):
    words_list = []
    words = word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    for word, tag in tagged_words:
        if tag == pos_tag:
            words_list.append(word)
    return words_list

get_words_by_pos_tag('NNP', article)

def get_words_by_multiple_pos_tags(pos_tags_list, text):
    words_list = []
    for tag in pos_tags_list:
        words_list.extend(get_words_by_pos_tag(tag, text))
    
    return list(set(words_list)) # in order to eliminate duplicates

tags = ['NN', 'VB']
get_words_by_multiple_pos_tags(tags, article)

# =====================================================================================================================================================
# Ex3

noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
nouns = get_words_by_multiple_pos_tags(noun_tags, article)
print(nouns)

verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
verbs = get_words_by_multiple_pos_tags(verb_tags, article)
print(verbs)

total_words = word_tokenize(article)
total_tagged_words = get_words_by_multiple_pos_tags(noun_tags + verb_tags, article)
percentage_total_tagged_words = (len(total_tagged_words) / len(total_words)) * 100
print(percentage_total_tagged_words)

# =====================================================================================================================================================
# Ex4

def print_table(N, text):
    print("{:<15} | {:<4} | {:<20} | {:<20}".format("Original word", "POS", "Simple lemmatization", "Lemmatization with POS"))
    print("-" * 80)
    symbols = ['``',"''",',', '.', ',','"',"'",'(',')',';',':','[',']','{','}','/','?','<','>','`','~','!','@','#','$','%','^','&','*','|',"\\",'-','_','=','+']
    lem = WordNetLemmatizer()
    distinct_results = set()
    sentences = sent_tokenize(text)
    for sentence in sentences[:N]:
        words = word_tokenize(sentence)
        words = list(filter(lambda x: x not in symbols, words))
        tagged_words = pos_tag(words)
        for word in tagged_words:
            original_word = word[0]
            pos = word[1][0].lower()
            try:
                simple_lemmatization = lem.lemmatize(original_word)
                if pos not in ['n', 'a', 'r', 'v']:
                    lemmatization_with_pos = lem.lemmatize(original_word)
                else:
                    lemmatization_with_pos = lem.lemmatize(original_word,pos)
            except:
                print(word)
            finally:
                if simple_lemmatization != lemmatization_with_pos and (original_word, pos) not in distinct_results:
                    distinct_results.add((original_word, pos))
                    print("{:<15} | {:<4} | {:<20} | {:<20}".format(original_word, pos if pos else '-', simple_lemmatization, lemmatization_with_pos))

print_table(20, article)

# =====================================================================================================================================================
# Ex5

def plot_graph(text):
    symbols = ['``',"''",',', '.', ',','"',"'",'(',')',';',':','[',']','{','}','/','?','<','>','`','~','!','@','#','$','%','^','&','*','|',"\\",'-','_','=','+']
    pos_counts = Counter()
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = list(filter(lambda x: x not in symbols, words))
        tagged_words = pos_tag(words)
        for word in tagged_words:
            pos = word[1]
            pos_counts[pos] += 1

    pos_counts = dict(sorted(pos_counts.items(), key=lambda item: item[1], reverse=True))

    plt.figure(figsize=(10, 6))
    plt.bar(pos_counts.keys(), pos_counts.values())
    plt.xlabel('Part of Speech')
    plt.ylabel('Number of Words')
    plt.title('Number of Words for Each Part of Speech')
    plt.xticks(rotation=45)
    plt.show()

plot_graph(article)

# =====================================================================================================================================================
# Ex6

gram=nltk.CFG.fromstring("""  
S -> NP VP 
NP -> DT NN | PRP
VP -> VBD | VBD NP PP SBAR
PP -> IN NP
SBAR -> IN S                         
DT -> 'The' | 'the'
NN -> 'cat' | 'bird' | 'garden'
VBD -> 'chased' | 'arrived'
IN -> 'in' | 'before'
PRP -> 'I'
""")
sentence = "The cat chased the bird in the garden before I arrived"
tokens = sentence.split()
rdp = nltk.RecursiveDescentParser(gram)
def print_trees_at_level(tree, level=0):
    if isinstance(tree, nltk.Tree):
        print(f"{'   ' * level}|->{tree.label()}")
        for subtree in tree:
            print_trees_at_level(subtree, level + 1)
    else:
        print(f"{'   ' * level}|->{tree}")

for tree in rdp.parse(tokens):
    print_trees_at_level(tree)

# =====================================================================================================================================================
# Ex7

gram=nltk.CFG.fromstring("""  
S -> NP VP 
NP -> DT NN | PRP
VP -> VBD | VBD NP PP SBAR
PP -> IN NP
SBAR -> IN S                         
DT -> 'The' | 'the'
NN -> 'cat' | 'bird' | 'garden'
VBD -> 'chased' | 'arrived'
IN -> 'in' | 'before'
PRP -> 'I'
""")
sentence = "The cat chased the bird in the garden before I arrived"
tokens = sentence.split()
rdp = nltk.RecursiveDescentParser(gram)
srp = nltk.ShiftReduceParser(gram)

rdp_trees = list(rdp.parse(tokens))
srp_trees = list(srp.parse(tokens))

def trees_are_equal(rdp_trees, srp_trees):
    for rdp_tree,srp_tree in zip(rdp_trees, srp_trees):
        if rdp_tree != srp_tree:
            print("The parsers produced different results.")
            return False
    print("The parsers produced the same results.")
    return True

trees_are_equal(rdp_trees, srp_trees)

gram2=nltk.CFG.fromstring("""  
S -> NP VP 
NP -> DT NN | DT NNS | NP PP
VP -> VBD PP PP
PP -> IN NP                       
DT -> 'The' | 'the' | 'a'
NN -> 'house' | 'crowbar' | 'night'
NNS -> 'burglars'
VBD -> 'broke'
IN -> 'into' | 'with' | 'during'
""")

sentence2 = "The burglars broke into the house with a crowbar during the night"
tokens2 = sentence2.split()
rdp2 = nltk.RecursiveDescentParser(gram2)
srp2 = nltk.ShiftReduceParser(gram2)

#rdp_trees2 = list(rdp2.parse(tokens2))
srp_trees2 = list(srp2.parse(tokens2))
#trees_are_equal(rdp_trees2, srp_trees2)

# In this case if we run rdp2.parse we break the kernel so this will count as diffrent results.
print("The parsers produced different results.")