import os
from nltk.parse.stanford import StanfordDependencyParser, StanfordParser

os.environ['JAVAHOME'] = 'C:\\Program Files\\Java\\jdk-20\\bin'
os.environ['STANFORD_PARSER'] = 'D:\\Master\\Anul1 Sem2\\NLP\\stanford-parser-full-2020-11-17\\stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'D:\\Master\\Anul1 Sem2\\NLP\\stanford-parser-full-2020-11-17\\stanford-parser-4.2.0-models.jar'


dep_parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
const_parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A curious cat explores the mysterious forest at night.",
    "Families gather around the dinner table, sharing stories.",
    "The sun shines brightly in the clear blue sky.",
    "The children play happily in the park under the warm afternoon sun."
]

output_file = "parsing_output.txt"

def parse_and_write_to_file(sentences, output_file):
    with open(output_file, 'w') as f:
        for i, sentence in enumerate(sentences, 1):
            f.write(f"Sentence - number {i}\n")
            f.write(f"{sentence}\n")
            
            constituency_parse = const_parser.raw_parse(sentence)
            f.write(str(list(constituency_parse)))
            f.write("\n")
            
            dependency_parse = dep_parser.raw_parse(sentence)
            for dep in dependency_parse:
                f.write(str(list(dep.triples())))
            f.write("\n")

            f.write("-" * 100)
            f.write("\n\n")

parse_and_write_to_file(sentences, output_file)