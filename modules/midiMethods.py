import pretty_midi
import numpy as np

# cc64 is sustain
# cc66 is sostenuto
# cc67 is soft

def handy_functions():
    #note or instrument names to numbers
    pretty_midi.note_name_to_number('C4')
    pretty_midi.instrument_name_to_program('Cello')
    
    # shift pitches of notes
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch += 5


def trim_silence(pm):
    "trims any silence from the beginning of a pretty midi object"
    
    # get the time of the first event (note or cc)
    if pm.instruments[0].control_changes[0] != []:
        delay = min(pm.instruments[0].notes[0].start, pm.instruments[0].control_changes[0])
    else:
        delay = pm.instruments[0].notes[0].start
    
    # subtract the delay from note objects
    for note in pm.instruments[0].notes:
        note.start = max(0, note.start - delay)
        note.end = max(0, note.end - delay)

    # subtract the delay from cc objects
    for cc in pm.instruments[0].control_changes:
        cc.time = max(0, cc.time - delay)


def sustain_only(pm):
    """Remove all non sustain cc messages from piano midi pm object"""
    filtered_cc = []
    # I'm going to assume that there is just one instrument, piano
    for cc in pm.instruments[0].control_changes:
        if cc.number == 64:
            filtered_cc.append(cc)
    pm.instruments[0].control_changes = filtered_cc


def pm_head(pm, seconds = 11):
    """Return the first seconds of a piano midi pm object"""
    pm.instruments[0].notes = [note for note in pm.instruments[0].notes if note.end < seconds]
    pm.instruments[0].control_changes = [cc for cc in pm.instruments[0].control_changes if cc.time < seconds]


def bin_sus(pm, cutoff = 50):
    """Set sustain so it is either completely on or completely off"""
    filtered_cc = []
    sustain = False
    # I'm going to assume that there is just one instrument, piano
    for cc in pm.instruments[0].control_changes:
        if cc.number == 64: # if it is sustain
            if sustain == False and cc.value >= cutoff:
                sustain = True
                cc.value = 127
                filtered_cc.append(cc)
            elif sustain == True and cc.value < cutoff:
                sustain = False
                cc.value = 0
                filtered_cc.append(cc)
        else:
            filtered_cc.append(cc)
    pm.instruments[0].control_changes = filtered_cc


def desus(pm, cutoff = 50):
    """Remove sustain pedal, and lengthen notes to emulate sustain effect"""
    # collect intervals in which pedal is down, and remove the pedal messages
    filtered_cc = []
    sustain = False
    intervals = []
    downtime = -1
    for cc in pm.instruments[0].control_changes:
        if cc.number == 64: # if it is sustain
            if sustain == False and cc.value >= cutoff:
                sustain = True
                downtime = cc.time
            elif sustain == True and cc.value < cutoff:
                sustain = False
                uptime = cc.time
                intervals.append((downtime, uptime))
        else:
            filtered_cc.append(cc)
    pm.instruments[0].control_changes = filtered_cc
    # print(intervals)

    # Now, use the intervals to extend out notes in them
    # We can structure our code like this because notes are ordered by end time
    # If that wasn't the case, we would need to do some sorting first
    index = 0
    last = 0
    extended_notes = []
    for note in pm.instruments[0].notes:
        while index < len(intervals) and note.end > intervals[index][1]:
            index += 1
        if index >= len(intervals):
            break
        # at this point, we know that note.end < intervals[index][1]
        # we test whether the end of the note falls in a sustain period
        if note.end > intervals[index][0] and note.end < intervals[index][1]:
            note.end = intervals[index][1]
            extended_notes.append(note)
        
    # now, we need to check for extended notes that have been extended over their compatriots...
    # this is horribly inefficient. But it does the job.
    # Could set it so comparisons are done between lists of same notes.
    for long_note in extended_notes:
        for note in pm.instruments[0].notes:
            if note.pitch == long_note.pitch and note.start < long_note.end and note.end > long_note.end:
                long_note.end = note.start
                # or could set it to note.end. I don't know which is best. Both seem ok.


def snap_to_grid(event_time, size=8):
    """takes an event time (in seconds) and gives it back snapped to a grid with 8ms between each event.
    I.e. multiples by 1000, then rounds to nearest multiple of 8
    
    Parameters
    ----------
    event_time : float
        Time of event, in seconds
    
    Returns
    ----------
    grid_time : int
        Time of event, in miliseconds, rounded to nearest 8 miliseconds

    """
    ms_time = event_time * 1000
    # get the distance to nearest number on the grid
    distance = ms_time % size
    if distance < size / 2:
        ms_time -= distance
    else:
        ms_time -= (distance - 8)
    return int(ms_time)


def pm2oore(pm):
    """Create event representation of midi. Must have sustain pedal removed.
    Will only have one note off for duplicate notes, even if multiple note offs are required.

    333 total possible events:
    0 - 87: 88 note on events, 
    88 - 175: 88 note off events
    176 - 300: 125 time shift events (8ms to 1sec)
    301 - 332: 32 velocity events

    Parameters:
    ----------
    pm : Pretty_Midi
        pretty midi object containing midi for a piano performance. Must have no sustain pedal.

    Returns:
    ----------
    events_with_shifts : list
        A list of events expressed as numbers between 0 and 332

    """
    # initially, store these in lists (time, int) tuples, where time is already snapped to 8ms grid,
    # and integers represent which event has taken place, from 0 to 332
    note_ons = []
    note_offs = []
    velocities = []
    n_velocities = 32
    for note in pm.instruments[0].notes:
        note_ons.append((snap_to_grid(note.start), note.pitch - 21)) # -21 because lowest A is note 21 in midi
        note_offs.append((snap_to_grid(note.end), note.pitch - 21 + 88))
        velocities.append((snap_to_grid(note.start), round(301 + note.velocity * (n_velocities - 1)/127))) #remember here we're mapping velocities to [0,n_velocities - 1]

    # remove duplicate consecutive velocities
    velocities.sort() #sort by time
    previous = (-1, -1)
    new_velocities = []
    for velocity in velocities:
        if velocity[1] != previous[1]: # check that we haven't just had this velocity
            new_velocities.append(velocity)
        previous = velocity
    velocities = new_velocities

    # Get all events, sorted by time
    # For simultaneous events, we want velocity change, then note offs, then note ons, so we
    # sort first by time, then by negative event number
    events = note_ons + note_offs + velocities
    events.sort(key = lambda x: (x[0], -x[1]))

    # add in time shift events. events 176 - 300.
    events_with_shifts = []
    previous_time = 0
    previous_event_no = -1
    for event in events:
        difference = event[0] - previous_time # time in ms since previous event
        previous_time = event[0] # update the previous event
        if difference != 0:
            shift = difference / 8 # find out how many 8ms units have passed
            seconds = int(np.floor(shift / 125)) # how many seconds have passed? (max we can shift at once)
            remainder = int(shift % 125) # how many more 8ms units do we need to shift?
            for seconds in range(seconds):
                events_with_shifts.append(300) # time shift a second
            if remainder != 0:
                events_with_shifts.append(remainder + 175)
        #append the event number only if it is not a repeated note off
        if 88 <= event[1] <= 175 and event[1] != previous_event_no or event[1] < 88 or event[1] > 175:
            events_with_shifts.append(event[1]) # append event no. only
            previous_event_no = event[1]
    return events_with_shifts        


def oore2midi(events):
    """Maps from a list of event numbers back to midi.

    333 total possible events:
    0 - 87: 88 note on events, 
    88 - 175: 88 note off events
    176 - 300: 125 time shift events (8ms to 1sec)
    301 - 332: 32 velocity events

    Parameters:
    ----------
    events_with_shifts : list
        A list of events expressed as numbers between 0 and 332

    Returns:
    ----------
    pm : Pretty_Midi
        pretty midi object containing midi for a piano performance.

    """
    pm = pretty_midi.PrettyMIDI(resolution=125)
    pm.instruments.append(pretty_midi.Instrument(0, name='piano'))

    notes_on = [] # notes for which there have been a note on event
    notes = [] # all the retrieved notes
    current_time = 0 # keep track of time (in seconds)
    current_velocity = 0

    for event in events:
        if 0 <= event <= 87:
            pitch = event + 21
            note = pretty_midi.Note(current_velocity, pitch, current_time, -1)
            notes_on.append(note)

        elif 88 <= event <= 175:
            end_pitch = event + 21 - 88
            new_notes_on = []
            for note in notes_on:
                if note.pitch == end_pitch:
                    note.end = current_time
                    notes.append(note)
                else:
                    new_notes_on.append(note)
            notes_on = new_notes_on
            
        elif 176 <= event <= 300:
            shift = event - 175
            current_time += (shift * 8 / 1000)

        elif 301 <= event <= 332:
            rescaled_velocity = np.round((event - 301) / 31 * 127)
            current_velocity = int(rescaled_velocity)
    notes.sort(key = lambda note: note.end)
    # Just in case there are notes for which note off was never sent, I'll clear notes_on
    
    pm.instruments[0].notes = notes
    return pm

def pitchM2pitchB(pitchM):
    """Maps midi notes to [0, 87]"""
    return pitchM - 21 # lowest A is 21


def ms2twinticks(time_ms, major=600, minor=10, max_major=9):
    """Maps ms to a major and a minor tick
    
    Arguments:
    time_ms -- int, time in ms to be mapped to bins
    major -- int, increments of major ticks
    minor -- int, increments of minor ticks

    Note:
    Assumes that max size of minor tick is one increment less than length of a major tick.

    Default bins work as follows:
    60 * 10ms minor ticks, 10 * 600ms major ticks, 5990ms possible
    590ms largest minor tick
    5400ms largest major tick
    Durations will likely be as follows:
    30 * 20ms, 18 * 600ms major ticks, 10780ms possible
    580ms largest minor tick
    10200ms largest major tick
    """

    # how many small bins filled, if we only had small bins?
    small_bins = round((time_ms / minor))
    # how many big bins are completely filled by these?
    big_bins = small_bins * minor // major
    # how many small bins are now left
    small_bins = small_bins - int((big_bins * major / minor))

    return (big_bins, small_bins)


def velM2velB(velocity, n_bins=32):
    """Maps midi velocity to binned velocity in [0, n_bins - 1]"""
    return round(velocity * (n_bins - 1)/127)




def pm2note_bin(pm):
    """Maps from pretty midi file to note_bin representation
    
    Arguments:
    pm -- pretty midi object to be converted

    Returns:
    note_bin -- list of shape (None, 6), where the sub lists contain
        [pitch, time shift major, time shift minor, duration major, duration minor, velocity]

    Note:
    Won't account for sustain pedal.

    """
    note_bin = []

    # need notes sorted by start time, not end time
    notes = sorted(pm.instruments[0].notes, key = lambda x: x.note.start)
    last_start = 0
    for note in notes:
        last_start = note.start
        # get the pitch bin number
        pitchB = pitchM2pitchB(note.pitch)

        # time shift: get the tuple of two bins
        shift = note.start - last_start
        shiftB = ms2twinticks(shift)

        # duration: get the tuple of two bins
        duration = note.end - note.start
        durationB = ms2twinticks(duration, major=600, minor=20, max_major=17)

        velocityB = velM2velB(note.velocity, n_bins=32)

        note_bin.append(pitchB, shiftB[0], shiftB[1], durationB[0], durationB[1], velocityB)
        
    return note_bin








    







class Tune:
    """Object for representing sequences of notes. Initializes from pretty midi object.
    
    Attributes:
    notes -- list of shape (None, 4), containing [time shift, pitch, velocity, duration]
    rep -- str, indicates representatation currently used

    """
    
    def __init__(self, pm, cc):
        self.notes = []
        self.rep = 0 # what representation is being used?
        # desus and extract notes from pm object
        for note in desus(pm).instruments[0].notes:
            self.notes.append([note.pitch, note.velocity, note.start, note.end])
        
        

def remap_velocity(notes):
    pass
    

def midi2notes(notes):
    """Converts midi to a note object based representation
    
    Arguments:
    pm -- pretty midi object to be converted
    
    Returns:
    notes --  list of shape (None, 4), containing (time shift, pitch, velocity, duration)
        where time shift
    
    Note: In the original paper, time shift has 13 major ticks and 77 minor ticks,
    representing 0 through 10 seconds.
    Duration has 25 major tick values and 40 minor ticks.  Here, we'll try leaving
    it as a continuous variable, and deal with that later. 
    
    """

    pass

def notes2midi():
    pass











