'''This module sets its own attributes based on the parsed input. 
For instance, if  "basis = cc-pvdz" is found in the input file,
then the attribute params.basis is set to "cc-pvdz".

All data is still stored in the global "options" dictionary
But module attributes are more convenient to type: 
     refer to: params.fragmentation
   instead of: params.options['fragmentation']
'''
import sys

options = {}
VERBOSE = False

def tryFloat(s):
    '''Try to cast to float, no big deal'''
    try:
        return float(s)
    except ValueError:
        return s

def parse(inFile):
    '''Crude input file parser. 
    
    Populates the "options" dictionary and sets own attributes
    Args:
        inFile: input file handle for reading
    Returns: 
        None
    '''
    global options

    options = {}
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
    
    thismodule = sys.modules[__name__]
    for option, value in options.items():
        if type(value) == str:
            if value == 'false' or value == 'no' or value == 'off':
                options[option] = False
            elif value == 'yes' or value == 'on' or value == 'true':
                options[option] = True
            elif len(value.split()) == 1:
                options[option] = tryFloat(value)
            else:
                options[option] = map(tryFloat, value.split())
        setattr(thismodule, option, value)
