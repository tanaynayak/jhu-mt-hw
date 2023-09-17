#!/usr/bin/env python
from itertools import islice
import sys
import optparse
from collections import defaultdict

SMOOTHING = 1e-10

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--iterations", dest="iterations", default=5, type="int", help="Number of iterations (default=5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=10, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

f_sentences = [line.strip().split() for line in open(f_data)]
e_sentences = [line.strip().split() for line in open(e_data)]

# Initial probabilities
t = defaultdict(float)  # Emission probabilities
q = defaultdict(float)  # Transition probabilities

# Initialization
for f_sentence, e_sentence in zip(f_sentences, e_sentences):
    for f_word in f_sentence:
        for e_word in e_sentence:
            t[(f_word, e_word)] = 1.0 + SMOOTHING

# Initialize transition probabilities
max_jump = 10
for jump in range(-max_jump, max_jump + 1):
    q[jump] = 1.0 / (2 * max_jump + 1) + SMOOTHING

for iter in range(opts.iterations):
    count_t = defaultdict(float)
    total_t = defaultdict(float)
    count_q = defaultdict(float)
    total_q = defaultdict(float)

    for f_sentence, e_sentence in islice(zip(f_sentences, e_sentences), opts.num_sents):
        for i, f_word in enumerate(f_sentence):
            Z = sum(t[(f_word, e_word)] * q[j-i] for j, e_word in enumerate(e_sentence))
            if Z == 0:
              Z = SMOOTHING
            for j, e_word in enumerate(e_sentence):
                c = t[(f_word, e_word)] * q[j-i] / Z
                count_t[(f_word, e_word)] += c
                total_t[e_word] += c
                
                jump = j - i
                count_q[jump] += c
                total_q[jump] += c

    for (f_word, e_word), v in count_t.items():
      total = total_t[e_word] if total_t[e_word] != 0 else SMOOTHING
      t[(f_word, e_word)] = v / total
    
    for jump, v in count_q.items():
      total = total_q[jump] if total_q[jump] != 0 else SMOOTHING
      q[jump] = v / total

# Output alignments using Viterbi
for f_sentence, e_sentence in islice(zip(f_sentences, e_sentences), opts.num_sents):
    best_alignment = []
    for i, f_word in enumerate(f_sentence):
        best_prob = 0
        best_j = -1
        for j, e_word in enumerate(e_sentence):
            prob = t[(f_word, e_word)] * q[j-i]
            if prob > best_prob:
                best_prob = prob
                best_j = j
        best_alignment.append((i, best_j))
    
    for (i, j) in best_alignment:
        sys.stdout.write("%i-%i " % (i, j))
    sys.stdout.write("\n")
