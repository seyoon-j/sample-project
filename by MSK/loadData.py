import pandas as pd
import numpy as np
from encoder import *


def load_file(file_path):
    whole_data = pd.read_excel(file_path)
    return whole_data


def format_data(file_path):
    data = load_file(file_path)

    vtg_d = data['VTG']
    vbg_d = data['VBG']
    vdd_d = data['VDD']
    idd_d = data['IDD']
    iss_d = data['ISS']
    itg_d = data['ITG']
    ibg_d = data['IBG']

    vtg_td = [tempstr.lstrip() for tempstr in vtg_d]
    vbg_td = [tempstr.lstrip() for tempstr in vbg_d]
    vdd_td = [tempstr.lstrip() for tempstr in vdd_d]
    idd_td = [tempstr.lstrip() for tempstr in idd_d]
    iss_td = [tempstr.lstrip() for tempstr in iss_d]
    itg_td = [tempstr.lstrip() for tempstr in itg_d]
    ibg_td = [tempstr.lstrip() for tempstr in ibg_d]

    input = [np.reshape([float(vtgstr[:vtgstr.find(' ')]) * checkVunit((vtgstr)),
                         float(vbgstr[:vbgstr.find(' ')]) * checkVunit(vbgstr),
                         float(vddstr[:vddstr.find(' ')]) * checkVunit(vddstr)], (3, 1))
             for vtgstr, vbgstr, vddstr in zip(vtg_td, vbg_td, vdd_td)]

    output = [np.reshape([float(iddstr[:iddstr.find(' ')]) * checkIunit(iddstr),
                          float(issstr[:issstr.find(' ')]) * checkIunit(issstr),
                          float(itgstr[:itgstr.find(' ')]) * checkIunit(itgstr),
                          float(ibgstr[:ibgstr.find(' ')]) * checkIunit(ibgstr)], (4, 1))
              for iddstr, issstr, itgstr, ibgstr in zip(idd_td, iss_td, itg_td, ibg_td)]

    return input, output


def checkVunit(vstring):
    return (" V" in vstring) + (1E-3 * ("mV" in vstring))


def checkIunit(istring):
    i_unit = (1E9 * (" A" in istring)) + (1E6 * ("mA" in istring)) + (1E3 * ("uA" in istring)) + (1E0 * ("nA" in istring)) \
             + (1E-3 * ("pA" in istring)) + (1E-6 * ("fA" in istring)) + (1E-9 * ("aA" in istring))
    # i_unit is multiplied by 1E9 to avoid using too small numbers
    return i_unit


def encoded_data(file_path):
    input_data, output_data = format_data(file_path)
    return encode(input_data, output_data)
