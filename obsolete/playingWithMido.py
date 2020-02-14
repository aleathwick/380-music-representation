import mido

filename = 'MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--4.midi'
mid = mido.MidiFile(filename)

#print(mid.tracks[0][1])
# <meta message time_signature numerator=4 denominator=4 clocks_per_click=24 notated_32nd_notes_per_beat=8 time=0>

print('\n'.join([str(message) for message in mid.tracks[1][:6]]))
del(mid.tracks[1][4])

print('new', '\n'.join([str(message) for message in mid.tracks[1][:6]]))

#msg.type gives you typ
#types are:
#track_name, program_change, control_change

#can also go msg.control, IF it is a control_change message
print(mido.backend)
print(mido.get_output_names())
print('\n'.join([str(msg.control) for msg in mid.tracks[1][:6] if msg.type == 'control_change']))

def remove_damper(mid):
    new_mid = mido.MidiFile()
    track = mido.MidiTrack()
    new_mid.tracks.append(track)
    #we need to iterate over tracks first, otherwise msg times will not be ints
    for track in mid.tracks:
        for msg in track:    
            if msg.type == 'control_change' and msg.control == 67:
                pass
            else:
                track.append(msg)
    return new_mid

new_mid = remove_damper(mid)
print('\n'.join([str(msg) for msg in mid.tracks[0][:6]]))
new_mid.save('new_song.mid')




