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
optparser.add_option("-t", "--iterations", dest="iterations", default=10, type="int", help="Number of iterations (default=10)")  # Increase default iterations
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000, type="int", help="Number of sentences to use for training and alignment")
opts, _ = optparser.parse_args()

f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

f_sentences = [line.strip().split() for line in open(f_data)]
e_sentences = [line.strip().split() for line in open(e_data)]

def train_hmm(f_sentences, e_sentences, iterations, num_sents):
    t = defaultdict(float)
    q = defaultdict(float)

    # Initialization
    for f_sentence, e_sentence in zip(f_sentences, e_sentences):
        for f_word in f_sentence:
            for e_word in e_sentence:
                t[(f_word, e_word)] = 1.0 + SMOOTHING

    max_jump = 10
    for jump in range(-max_jump, max_jump + 1):
        q[jump] = 1.0 / (2 * max_jump + 1) + SMOOTHING

    for iter in range(iterations):
        count_t = defaultdict(float)
        total_t = defaultdict(float)
        count_q = defaultdict(float)
        total_q = defaultdict(float)

        for f_sentence, e_sentence in islice(zip(f_sentences, e_sentences), num_sents):
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
    
    return t, q

def align_hmm(t, q, f_sentences, e_sentences, num_sents):
    alignments = []

    for f_sentence, e_sentence in islice(zip(f_sentences, e_sentences), num_sents):
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
        alignments.append(best_alignment)

    return alignments

# Train in both directions
t_f2e, q_f2e = train_hmm(f_sentences, e_sentences, opts.iterations, opts.num_sents)
t_e2f, q_e2f = train_hmm(e_sentences, f_sentences, opts.iterations, opts.num_sents)

alignments_f2e = align_hmm(t_f2e, q_f2e, f_sentences, e_sentences, opts.num_sents)
alignments_e2f = align_hmm(t_e2f, q_e2f, e_sentences, f_sentences, opts.num_sents)

# Symmetrization
for (align_f2e, align_e2f) in zip(alignments_f2e, alignments_e2f):
    combined_alignment = set(align_f2e)
    for (j, i) in align_e2f:
        combined_alignment.add((i, j))
    sys.stdout.write(" ".join(["%d-%d" % (i, j) for (i, j) in combined_alignment]) + "\n")
