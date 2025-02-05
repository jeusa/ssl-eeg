import eel
import random
import time
import json

import util

cur_n = -1
seq, targets, score, block_seq = [], [], [], []
all_blocks = {}

def generate_sequence(n):
    n = int(n)
    global score, seq, targets, cur_n
    cur_n = n
    targets = random.sample(range(n, util.trial_length), k=int(util.target_share*util.trial_length))
    targets.sort()

    seq = []

    if n > 0:
        for i in range(util.trial_length):

            if i in targets:
                cur = seq[i-n]
            else:
                s = util.symbols.copy()
                if i-n >= 0:
                    s.remove(seq[i-n])
                cur = random.choice(s)

            seq.append(cur)

    else:
        s = util.symbols.copy()
        s.remove("x")
        for i in range(util.trial_length):

            if i in targets:
                cur = "x"
            else:
                cur = random.choice(s)

            seq.append(cur)

    score = [None] * util.trial_length

def generate_block_sequence(not_start=None):

    bseq = list(range(0, 4))
    random.shuffle(bseq)

    if not not_start==None:
        while bseq[0]==not_start:
            random.shuffle(bseq)

    return bseq

def generate_session_block_sequence():
    bseq = generate_block_sequence()
    bseq += generate_block_sequence(bseq[-1])

    global block_seq
    block_seq = bseq

def save_block():
    block_data = {
    "n": cur_n,
    "sequence": seq,
    "score": score,
    "targets": targets
    }

    global all_blocks
    all_blocks[f"block_{len(all_blocks)}"] = json.dumps(block_data)

    with open("n_back_session.json", "w") as sf:
        print(all_blocks, file=sf)
