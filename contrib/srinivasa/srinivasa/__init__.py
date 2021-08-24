from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, \
    Number, Operator, Whitespace, Text


# This style is based on the "stata" style

class SrinivasaStyle(Style):
    default_style = ''
    styles = {
        Text:                  '#000000',
        Whitespace:            '#bbbbbb',
        Error:                 'bg:#e3d2d2 #a61717',
        String:                '#7a2424',
        Number:                '#003399',
        Operator:              '',
        Name.Function:         '#2c2cff',
        Name.Other:            '#be646c',
        Keyword:               'bold #003399',
        Keyword.Constant:      '',
        Comment:               'italic #2E8B57', # SeaGreen
        Name.Variable:         'bold #2E8B57', # SeaGreen
        Name.Variable.Global:  'bold #b5565e',
    }
