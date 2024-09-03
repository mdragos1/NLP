from nltk.corpus import wordnet as wn

#1
def print_glosses(word):
    synsets = wn.synsets(word)
    print(f"Glosses for all senses of the word '{word}':")
    for synset in synsets:
        print(f"{synset.name()}: {synset.definition()}")

#print_glosses('phone')

#2
def check_synonyms(word1, word2):
    synsets1 = wn.synsets(word1)
    for synset1 in synsets1:
        for lemma in synset1.lemmas():
            if word2 in lemma.name():
                print(f"Synonym found in synset '{synset1.name()}': {synset1.definition()}")
                return True
    synsets2 = wn.synsets(word2)
    for synset2 in synsets2:
        for lemma in synset2.lemmas():
            if word1 in lemma.name():
                print(f"Synonym found in synset '{synset2.name()}': {synset2.definition()}")
                return True
    print(f"No common synset found for words '{word1}' and '{word2}'.")
    return False

#check_synonyms('good', 'well')

#3
def get_holonyms_meronyms(synset):
    holonyms = synset.member_holonyms() + synset.part_holonyms() + synset.substance_holonyms()
    meronyms = synset.member_meronyms() + synset.part_meronyms() + synset.substance_meronyms()
    return holonyms, meronyms

def print_holonyms_meronyms(word):
    synset = wn.synsets(word)
    holonyms, meronyms = get_holonyms_meronyms(synset[0])
    
    print(f"Holonyms for '{word}':")
    for holonym in holonyms:
        print(f"{holonym.name()}")
    
    print(f"Meronyms for '{word}':")
    for meronym in meronyms:
        print(f"{meronym.name()}")

#print_holonyms_meronyms('tree')


#4
def print_hypernym_path(synset):
    hypernyms = synset.hypernym_paths()
    for path in hypernyms:
        print(f"Hypernym path for '{synset.name()}':")
        for hypernym in path:
            print(f"{hypernym.name()}")

#print_hypernym_path(wn.synsets('dog')[0])


#5
def common_hypernyms_min_path(synset1, synset2):
    hypernyms1 = {synset: i for i, synset in enumerate(synset1.hypernym_paths()[0])}
    hypernyms2 = {synset: i for i, synset in enumerate(synset2.hypernym_paths()[0])}
    
    common_hypernyms = set(hypernyms1.keys()).intersection(hypernyms2.keys())
    min_path_sum = float('inf')
    min_hypernyms = []
    
    for hypernym in common_hypernyms:
        path_sum = hypernyms1[hypernym] + hypernyms2[hypernym]
        if path_sum < min_path_sum:
            min_path_sum = path_sum
            min_hypernyms = [hypernym]
        elif path_sum == min_path_sum:
            min_hypernyms.append(hypernym)
    
    return min_hypernyms

dog_synset = wn.synsets('dog')[0]
cat_synset = wn.synsets('cat')[0]
min_hypernyms = common_hypernyms_min_path(dog_synset, cat_synset)
print(f"Common hypernyms with minimum path sum for '{dog_synset.name()}' and '{cat_synset.name()}':")
for hypernym in min_hypernyms:
    print(f"{hypernym.name()}")


#6
def sort_by_similarity(synset, synsets):
    similarities = [(other, synset.wup_similarity(other)) for other in synsets]
    sorted_synsets = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_synsets

synset = wn.synsets('cat')[0]
synsets = [wn.synsets(word)[0] for word in ['animal', 'tree', 'house', 'object', 'public_school', 'mouse']]
# sorted_synsets = sort_by_similarity(synset, synsets)

# print(f"Synsets sorted by similarity to '{synset.name()}':")
# for syn, sim in sorted_synsets:
#     print(f"{syn.name()} (similarity: {sim})")


#7
def check_indirect_meronyms(synset1, synset2, hypernym):
    meronyms1 = set(synset1.closure(lambda s: s.member_meronyms() + s.part_meronyms() + s.substance_meronyms()))
    meronyms2 = set(synset2.closure(lambda s: s.member_meronyms() + s.part_meronyms() + s.substance_meronyms()))
    return hypernym in meronyms1 and hypernym in meronyms2

synset1 = wn.synset('wheel.n.01')
synset2 = wn.synset('seat.n.01')
hypernym = wn.synset('car.n.01')
# indirect_meronyms = check_indirect_meronyms(synset1, synset2, hypernym)
# print(f"Are '{synset1.name()}' and '{synset2.name()}' indirect meronyms of '{hypernym.name()}': {indirect_meronyms}")


#8
def print_synonyms_antonyms(adjective):
    synsets = wn.synsets(adjective, pos=wn.ADJ)
    for synset in synsets:
        synonyms = {lemma.name() for lemma in synset.lemmas()}
        antonyms = {ant.name() for lemma in synset.lemmas() for ant in lemma.antonyms()}
        print(f"Sense '{synset.name()}': {synset.definition()}")
        print(f"  Synonyms: {synonyms}")
        print(f"  Antonyms: {antonyms}")

#print_synonyms_antonyms('beautiful')