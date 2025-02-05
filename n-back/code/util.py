import eel
from pylsl import StreamInfo, StreamOutlet

# n-back parameters
symbols = list("bcdfghjklmnpqrstvwxyz")
trial_length = 30
target_share = 1/3

time_symbol = 1000
time_pause = 2000
time_session_pause = 10000
time_n_info = 5000

# set up LabStreamingLayer stream
info = StreamInfo(name='n-back', type='Markers', channel_count=1, nominal_srate=0, channel_format='string', source_id='n-back-marker')
outlet = StreamOutlet(info)

# lsl marker strings
lsl_start_session = "start_session"
lsl_start_block = "start_block"
lsl_n = "n="
lsl_end_block = "end_block"
lsl_end_session = "end_session"


@eel.expose
def set_session_parameters():
    eel.setTimeParameters(time_session_pause, time_n_info)

@eel.expose
def set_block_parameters():
    eel.setTimeParameters(time_symbol, time_pause)


def send_marker(marker):
    outlet.push_sample([marker])
