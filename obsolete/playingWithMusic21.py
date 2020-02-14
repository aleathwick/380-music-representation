from music21 import converter, instrument, midi, note, pitch

filename = 'tf.midi'
filename2 = 'MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--4.midi'

#Create a midi object, and open/read in a file
mf = midi.MidiFile()
mf.open(filename)
mf.read()
mf.close()

def print_events(mf, n=15):
    """print the first n events for each track"""
    for i in range(len(mf.tracks)):
        print("length of track: ", len(mf.tracks[i].events))
        print("track: ", i)
        print("beginning:")
        print('\n'.join([str(msg) for msg in mf.tracks[i].events[:n]]))
        print("end:") 
        print('\n'.join([str(msg) for msg in mf.tracks[i].events[-n:]]))

def desoft(mf):
    """Remove messages controlling soft (67) and sostenuto (66) pedals, and merge adjacent deltatime messages""" 
    for i in range(len(mf.tracks)): # iterate through the tracks
        filtered_events = [] # create new events list
        for msg in mf.tracks[i].events: # iterate through the events
            if msg.type == 'CONTROLLER_CHANGE' and msg.parameter1 == 66 or msg.parameter1 == 67:
                pass
            # If the next message is DeltaTime, and so is the last one appended, then we need to merge them
            elif msg.isDeltaTime() and filtered_events != [] and filtered_events[-1].type == 'DeltaTime':
                filtered_events[-1].time += msg.time
            # Otherwise, we can just add the message in
            else:
                filtered_events.append(msg)
        mf.tracks[i].events = filtered_events

def bin_sus(mf, cutoff=90):
    """set sustain (64) to work in either on or off fashion"""
    sustain = False # records current pedal position
    filtered_events = [] # create new events list
    for msg in mf.tracks[1].events: # iterate through the events
        if msg.type == 'CONTROLLER_CHANGE' and msg.parameter1 == 64:
            if sustain == False and msg.parameter2 <= cutoff: # is sustain off, but the pedal going beneath threshold?
                sustain = True
                msg.parameter2 = 0
                filtered_events.append(msg)
            elif sustain == True and msg.parameter2 > cutoff: # vice versa?
                sustain = False
                msg.parameter2 = 127
                filtered_events.append(msg)
            else: # ignore all other sustain messages
                pass
        # If the next message is DeltaTime, and so is the last one appended, then we need to merge them
        elif msg.isDeltaTime() and filtered_events != [] and filtered_events[-1].type == 'DeltaTime':
            filtered_events[-1].time += msg.time
        # Otherwise, we can just add the message in
        else:
            filtered_events.append(msg)
    mf.tracks[1].events = filtered_events

# def desus(mf, cutoff = 90):
#     """Remove sustain and instead make notes longer"""

#     for i in range(len(mf.tracks)): # iterate through the tracks
#         sustain = False
#         note_off_stack = []
#         filtered_events = [] # create new events list
#         for msg in mf.tracks[i].events: # iterate through the events
#             if msg.type == 'CONTROLLER_CHANGE' and msg.parameter1 == 64:
#                 if sustain == False and msg.parameter2 <= cutoff: # is sustain off, but the pedal going beneath threshold?
#                     sustain = True
#                 elif sustain == True and msg.parameter2 > cutoff: # vice versa?
#                     sustain = False
#                     while note_off_stack != []:
#                         if 
#                         dt = midi.DeltaTime(msg.track)
#                         dt.time = 0
#                         #filtered_events.append(dt) # append a delta event
#                         note_off = note_off_stack.pop()
#                         #filtered_events.append(note_off) # append the note off event
#                         #print(note_off)
#                 else: # ignore all other sustain messages
#                     pass
#             # If the next message is DeltaTime, and so is the last one appended, then we need to merge them
#             elif msg.type == 'DeltaTime' and filtered_events != [] and filtered_events[-1].isDeltaTime():
#                 filtered_events[-1].time += msg.time
#             # If the message is a note off event, and pedal is down, we need to delay adding it
#             elif msg.type == 'NOTE_ON' and msg.velocity == 0:
#                 note_off_stack.append(msg)
#             else:
#                 filtered_events.append(msg)
#         mf.tracks[i].events = filtered_events

def desus2(mf, cutoff = 90):
    """Remove sustain and instead make notes longer. Needs already"""
    sustain = False
    note_off_stack = []
    filtered_events = [] # create new events list
    for msg in mf.tracks[1].events: # iterate through the events
        if msg.type == 'CONTROLLER_CHANGE' and msg.parameter1 == 64:
            if msg.parameter2 == 127 and not sustain: # is sustain off, but the pedal going beneath threshold?
                sustain = True
            elif msg.parameter2 == 0 and sustain: # vice versa?
                sustain = False
                while note_off_stack != []:
                    note_off = note_off_stack.pop()
                    filtered_events.append(note_off) # append the note off event
                    dt = midi.DeltaTime(msg.track)
                    dt.time = 0
                    filtered_events.append(dt) # append a delta event
        # If the next message is DeltaTime, and so is the last one appended, then we need to merge them
        elif msg.type == 'DeltaTime' and filtered_events != [] and filtered_events[-1].type == 'DeltaTime':
            filtered_events[-1].time += msg.time
        # If the message is a note off event, and pedal is down, we need to delay adding it
        elif msg.type == 'NOTE_ON' and msg.velocity == 0:
            note_off_stack.append(msg)
            #print(msg)
        else:
            filtered_events.append(msg)
    mf.tracks[1].events = filtered_events

def desus3(mf, cutoff = 90):
    """Remove sustain and instead make notes longer. Needs already"""
    sustain = False
    note_off_stack = []
    filtered_events = [] # create new events list
    for msg in mf.tracks[1].events: # iterate through the events
        if msg.type == 'CONTROLLER_CHANGE' and msg.parameter1 == 64:
            if msg.parameter2 == 127 and not sustain: # is sustain off, but the pedal going beneath threshold?
                sustain = True
            elif msg.parameter2 == 0 and sustain: # vice versa?
                sustain = False
                while note_off_stack != []:
                    if filtered_events != [] and not filtered_events[-1].isDeltaTime() or filtered_events == []:
                        dt = midi.DeltaTime(msg.track)
                        #dt = midi.DeltaTime(note_off_stack[-1].track)
                        dt.time = 0
                        filtered_events.append(dt) # append a delta event
                    note_off = note_off_stack.pop()
                    note_off.track = msg.track
                    filtered_events.append(note_off) # append the note off event
                    #print(note_off)
        # If the next message is DeltaTime, and so is the last one appended, then we need to merge them
        elif msg.type == 'DeltaTime' and filtered_events != [] and filtered_events[-1].type == 'DeltaTime':
            filtered_events[-1].time += msg.time
        # If the message is a note off event, and pedal is down, we need to delay adding it
        elif sustain and msg.type == 'NOTE_ON' and msg.velocity == 0:
            note_off_stack.append(msg)
            #print(msg)
        else:
            filtered_events.append(msg)
    mf.tracks[1].events = filtered_events

def one_track(mf):
    pass


def filter_midi1(mf, filter_type='CONTROLLER_CHANGE', filter_parameter1=67):
    """Remove messages of message type and filter_parameter""" 
    for i in range(len(mf.tracks)):
        filtered_events = []
        for msg in mf.tracks[i].events:
            if msg.type == filter_type: #and msg.parameter1 == filter_parameter:
                pass
            else:
                filtered_events.append(msg)
        mf.tracks[i].events = filtered_events




def save_midi(mf, filename='new_midi.midi'):
    """save midi to file"""
    mf.open(filename, attrib='wb')
    mf.write()


# dt = midi.DeltaTime()
# print(dt)



print('############################### before ##################################')

print_events(mf, n=300)
mf.tracks[1]
desus3(mf)
print('############################### after ###################################')
print_events(mf, n=300)
save_midi(mf, filename='21desus2.midi')

# mf.tracks and mf.tracks[i].events are just lists

# # Handy attributes midi messages have:
# msg.type
# msg.data
# msg.pitch
# msg.velocity

# #Some handy methods that midi messages have:
# msg.isNoteOn()
# msg.isNoteOff()
# msg.isDeltaTime()
# msg.matchedNoteOff(other_msg)





