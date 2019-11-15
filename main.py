import numpy as np
import random

def shallow_flatten(L):
    for e in L:
        if type(e) == list:
            yield from e
        else:
            yield e


def group_ngrams(L, n):
    output = []
    for i in range(0, len(L), n):
        output += L[i:i + n]
    return output


class Node:
    def __init__(self, chain, word, ID):
        self.chain = chain
        self.word = word
        self.id = ID
        self.transitions = [0 if i != self else 1 for i in self.chain.nodes]
    
    
    def set_probabilities(self):
        sum_probs = sum(self.transitions)
        for i in range(len(self.transitions)):
            self.transitions[i] /= sum_probs


class MarkovChain:
    def __init__(self):
        self.nodes = {}
    

    def add_node(self, word):
        self.nodes[word] = Node(self, word, len(self.nodes))
        for node in self.nodes.items():
            node[1].transitions.append(0)
    

    def train(self, file_name):
        # Get text.
        with open(file_name, "r") as rf:
            # Text processing steps.
            text_lines = [lines.strip() for lines in rf.readlines()]
            # Add line break characters at line breaks
            # so the chain can generate newlines as well.
            text = group_ngrams(" ".join(text_lines).split(" "), 4)
            # Separate punctuation marks.
            punctuation = [".", ",", "!", "?", ":", ";", "..."]
            for i in range(len(text)):
                if text[i] != "":
                    if text[i][-1] in punctuation:
                        text[i] = [text[i][:-1], text[i][-1]]
                    elif text[i][-1] == "\"":
                        text[i] = text[i][:-1]
                    elif text[i][0] == "\"":
                        text[i] = text[i][1:]
            text = list(shallow_flatten(text))

        self.add_node(text[0])
        prev_state = self.nodes[text[0]]
        for i in range(1, len(text)):
            try:
                curr_state = self.nodes[text[i]]
            except KeyError:
                self.add_node(text[i])
                curr_state = self.nodes[text[i]]
            prev_state.transitions[curr_state.id] += 1
            prev_state = curr_state
        
        for node in self.nodes.items():
            node[1].set_probabilities()
    

    def generate_text(self, length):
        text = []
        curr_state = self.nodes[random.choice(list(self.nodes.keys()))]
        text.append(curr_state.word)

        ctr = 0
        punctuation = [".", ",", ":", ";", "!", "?", "..."]
        while ctr < length:
            curr_state = self.nodes[np.random.choice(list(self.nodes.keys()), 1, p = curr_state.transitions).item()]
            text.append(" " + curr_state.word if not curr_state.word in punctuation else curr_state.word)
            if curr_state.word == ".":
                ctr += 1
        
        return "".join(text)
        

if __name__ == "__main__":
    chain = MarkovChain()
    chain.train("beemovie.txt")
    print(chain.generate_text(3))

