from nltk.tokenize import word_tokenize
import nltk


prod_rules = """S -> NP VP
S -> VP
NP -> DT NN
NP -> DT JJ NN
NP -> PRP
VP -> VBP NP
VP -> VBP VP
VP -> VBG NP
VP -> TO VP
VP -> VB
VP -> VB NP
NN -> "show" | "book"
PRP -> "I"
VBP -> "am"
VBG -> "watching"
VB -> "show"
DT -> "a" | "the"
MD -> "will"
"""
grammar = nltk.CFG.fromstring(prod_rules)
for rule in grammar.productions():
    print(rule.lhs(), rule.rhs())

prod_rules = prod_rules.splitlines()
rules_dict = dict()

for rule in prod_rules:
    symbol, left_corner_rule = rule.split(sep=" -> ")
    left_corner_rule = left_corner_rule.split(sep=" | ")
    for lcr in left_corner_rule:    
        if symbol in rules_dict.keys():
            if(lcr.split()[0] not in rules_dict[symbol]):
                rules_dict[symbol].append(lcr.split()[0])
        else:
            rules_dict[symbol] = [lcr.split()[0]]

for k,v in zip(rules_dict.keys(), rules_dict.values()):
    print(f"{k}: {v}")

def get_keys_from_value(d:dict, val):
    return [k for k, v in d.items() if any(value == val for value in v)][0]

a = rules_dict.items()
sentence = "I am watching a show"

def find_next_non_terminal(key, current_word):
    nt = []
    for r in grammar.productions():
        non_terminal = r.lhs()._symbol
        try:
            symbols = [s._symbol for s in r.rhs()]
        except:
            break
        if key == non_terminal and current_word in symbols:
            l = [s for s in symbols if s != current_word]
            if len(l) > 0:
                nt.append(l)
    return nt

def parse(sentence, start_symbol='S'):
    parse_tree = []
    words = word_tokenize(sentence)
    new_words = []
    for w in words:
        new_words.append(f'"{w}"')
    words = new_words    
    non_terminal = start_symbol
    processed_words = []
    total_non_terminal = [start_symbol]
    non_terminal_is_bad = False
    while non_terminal is not None:
        if words:
            current_word = words[0]
            current_key = get_keys_from_value(rules_dict, current_word)
            current_tree = []
            while current_key != non_terminal:
                if non_terminal in rules_dict and len(current_key) > 0:
                    current_tree.append((current_key, current_word))   
                    current_word = current_key
                    try:
                        current_key = get_keys_from_value(rules_dict, current_word)
                    except:
                        non_terminal_is_bad = True
                        break
                    
                else:
                    break
            if non_terminal_is_bad:
                non_terminal_is_bad = False
                parse_tree.pop()
                try:
                    non_terminal = possible_non_terminal_list[0].pop(0)
                    total_non_terminal.pop()
                    parse_tree.append((total_non_terminal[-1],non_terminal))
                    total_non_terminal.append(non_terminal)
                except:
                    possible_non_terminal_list.pop(0)
                    non_terminal = possible_non_terminal_list[0].pop(0)
                    total_non_terminal.pop()
                    parse_tree.append((total_non_terminal[-1],non_terminal))
                    total_non_terminal.append(non_terminal)
            else:
                k,w = current_key, current_word
                parse_tree.append((k, w))
                current_tree.reverse()
                parse_tree.extend(current_tree)
                possible_non_terminal_list = find_next_non_terminal(k,w)
                try:
                    new_non_terminal = possible_non_terminal_list[0].pop(0)
                    parse_tree.append((non_terminal, new_non_terminal))
                    non_terminal = new_non_terminal
                except:
                    processed_words.append(words.pop(0))
                    break
                total_non_terminal.append(non_terminal)
                processed_words.append(words.pop(0))
        else:
            break

    if len(words) == 0:
        return parse_tree
    else:
        return None
    
parse_tree = parse(sentence)
print(parse_tree)