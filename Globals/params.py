'''This module contains the globally-shared options dict and a method to
generate it by parsing an input file. Other modules set module-level attributes
like "quiet" and "verbose".
'''
options = {}
verbose = False
qm_logfile = None

def convert_params():
    '''Sanitize parsed options.

    Make string-->(list, float, boolean) conversions, wherever possible
    '''
    global options
    for option, value in options.items():
        if type(value) == str:
            if value == 'false' or value == 'no' or value == 'off':
                options[option] = False
            elif value == 'yes' or value == 'on' or value == 'true':
                options[option] = True
            elif len(value.split()) == 1:
                options[option] = tryFloat(value.split()[0])
            else:
                options[option] = map(tryFloat, value.split())

def tryFloat(s):
    '''Try to cast to float; no big deal'''
    try:
        return float(s)
    except ValueError:
        return s

def parse(inFile):
    '''Crude input file parser.

    Populates the "options" dictionary from an input file.

    Args:
        inFile: input file handle for reading
    Returns:
        None
    '''
    global options

    options = {}
    options['input_filename'] = inFile.name
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
            if key in options:
                raise RuntimeError("%s double specified" % key)
            options[key] = value

        # multi-line entry, enclosed in { }
        elif '{' in line:
            key = line.split('{')[0].strip().lower()
            if key:
                if key in options:
                    raise RuntimeError("%s double specified" % key)
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
    convert_params()
