import nltk

def get_processed_grammar(grammar: nltk.CFG, grammar_str: str):
    index = 1
    modified = True
    while modified:
        modified = False
        for p in grammar.productions():
            if len(p.rhs()) > 2:
                l = [str(i) for i in p._rhs]
                target = ' '.join(l) + '\n'
                replacement = f'{str(l[0])} NewT{index}\n    NewT{index} -> '+' '.join(l[1:])+'\n'
                new_grammar_str = grammar_str.replace(target, replacement)
                if grammar_str == new_grammar_str:
                    target = ' '.join(l)
                    replacement = f'{str(l[0])} NewT{index}'
                    add = f'\nNewT{index} -> ' + ' '.join(l[1:])+'\n'
                    new_grammar_str = grammar_str.replace(target, replacement)
                    new_grammar_str  = new_grammar_str.__add__(add)
                grammar = nltk.CFG.fromstring(new_grammar_str)
                index += 1
                modified = True
                break
    return grammar

def parse_sentence(sentence:str, grammar:nltk.CFG) -> bool:
    words = sentence.split()
    n = len(words)
    
    T = [[None] * (n+1) for _ in range(n+1)]
    
    for i in range(n):
        productions = grammar.productions(rhs=words[i])
        T[i][i+1] = productions[0].lhs()
    
    index = dict((p.rhs(), p.lhs()) for p in grammar.productions())
    n = n
    for span in range(2, n+1):
        for i in range(n+1-span):
            j = i + span
            for k in range(i+1, j):
                left, right = T[i][k], T[k][j]
                if left and right and (left, right) in index:
                    T[i][j] = index[(left, right)]

    
    if nltk.Nonterminal('S') == T[0][n]:
        return True, T
    else:
        return False, T

def print_matrix(T: list[list]):
    for line in T:
        l = ['None' if el is None else str(el) for el in line]
        print('   '.join(l))

grammar_str = """
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
"""

grammar = nltk.CFG.fromstring(grammar_str)
grammar = get_processed_grammar(grammar, grammar_str)
sentence = "I shot an elephant in my pajamas"
parsed, T = parse_sentence(sentence, grammar)
if parsed:
    print_matrix(T)
else:
    print_matrix(T)
