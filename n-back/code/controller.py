import eel

import n_back
import util

@eel.expose
def start_session():
    n_back.generate_session_block_sequence()

    util.send_marker(util.lsl_start_session)

@eel.expose
def end_session():
    util.send_marker(util.lsl_end_session)

@eel.expose
def start_n_back(n):
    n_back.generate_sequence(n)
    eel.showSequence(n_back.seq)

    util.send_marker(util.lsl_start_block)
    util.send_marker(util.lsl_n + str(n))

@eel.expose
def end_n_back():
    util.send_marker(util.lsl_end_block)
    n_back.save_block()

@eel.expose
def get_next_n():
    n = -1

    if len(n_back.block_seq)>0:
        n = n_back.block_seq.pop(0)

    eel.setNextN(n)

@eel.expose
def target_pressed(idx):
    idx = int(idx)

    if n_back.score[idx] == None:
        if idx in n_back.targets:
            n_back.score[idx] = True
            eel.correctInputFeedback()
        else:
            n_back.score[idx] = False
            eel.wrongInputFeedback()

@eel.expose
def no_target_pressed(idx):
    idx = int(idx)

    if n_back.score[idx] == None:
        if idx not in n_back.targets:
            n_back.score[idx] = True
            eel.correctInputFeedback()
        else:
            n_back.score[idx] = False
            eel.wrongInputFeedback()
