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
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

f_sentences = [['NULL'] + line.strip().split() for line in open(f_data)]
e_sentences = [line.strip().split() for line in open(e_data)]

alignments = defaultdict(list)

def train_model(f_sentences, e_sentences):
    t = defaultdict(float)
    q = defaultdict(float)
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

    return t, q

t_fe, q_fe = train_model(f_sentences, e_sentences)
t_ef, q_ef = train_model(e_sentences, f_sentences)

# Using alignments in both directions and symmetrize
for f_sentence, e_sentence in islice(zip(f_sentences, e_sentences), opts.num_sents):
    alignments_fe = []
    alignments_ef = []
    for i, f_word in enumerate(f_sentence):
        best_prob_fe = 0
        best_j_fe = -1
        best_prob_ef = 0
        best_j_ef = -1
        for j, e_word in enumerate(e_sentence):
            prob_fe = t_fe[(f_word, e_word)] * q_fe[j-i]
            prob_ef = t_ef[(e_word, f_word)] * q_ef[i-j]
            if prob_fe > best_prob_fe:
                best_prob_fe = prob_fe
                best_j_fe = j
            if prob_ef > best_prob_ef:
                best_prob_ef = prob_ef
                best_j_ef = j
        alignments_fe.append((i, best_j_fe))
        alignments_ef.append((best_j_ef, i))

    # Intersect
    intersected = [pair for pair in alignments_fe if pair in alignments_ef]

    # Union
    unioned = list(set(alignments_fe + alignments_ef))

    # Output the alignments; currently using intersected; change to unioned if required
    for (i, j) in intersected:
        sys.stdout.write("%i-%i " % (i, j))
    sys.stdout.write("\n")
