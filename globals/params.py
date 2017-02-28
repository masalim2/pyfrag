# Globally-shared data
options = {}
VERBOSE = False

def parse(inFile):
    '''Crude input file parser. 
    Returns dictionary of calculation attributes'''

    inputLines = inFile.readlines()
    nlines = len(inputLines)
    n = 0
    while n < nlines:
        n2 = n + 1
        line = inputLines[n].split('#')[0]

        # single-line (key) = (value) entries
        entry = [s.strip().lower() for s in line.split('=')]
        if len(entry) == 2 and '' not in entry:
            key, value = entry
            options[key] = value

        # multi-line entry, enclosed in { }
        elif '{' in line:
            key = line.split('{')[0].strip().lower()
            if key:
                options[key] = []
                while n2 < nlines:
                    closer = False
                    line2 = inputLines[n2].split('#')[0]
                    if '}' in line2:
                        closer = True
                    line2 = line2.split('}')[0].strip().lower()
                    if line2:
                        options[key].append(line2)
                    n2 += 1
                    if closer:
                        break
        n = n2
    
    for option, default in defaults.items():
        if option not in options:
            options[option] = default

    for option, value in options.items():
        if type(value) == str:
            if value == 'false' or value == 'no' or value == 'off':
                options[option] = False
            if value == 'yes' or value == 'on' or value == 'true':
                options[option] = True
