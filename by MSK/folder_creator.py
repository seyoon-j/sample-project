import pickle
import time
import os


def date_string():
    datetime_strct = time.localtime()
    folder_string = f"{datetime_strct.tm_year}-{datetime_strct.tm_mon}-" \
                    f"{datetime_strct.tm_mday}_{datetime_strct.tm_hour}-" \
                    f"{datetime_strct.tm_min}-{datetime_strct.tm_sec}"
    return folder_string


def mainfolder():
    folder_string = date_string()
    origin_dir = os.getcwd()
    os.mkdir(f"{origin_dir}/{folder_string}")
    os.chdir(f"{origin_dir}/{folder_string}")


def subfolders(run_number, variable, variable_name):
    origin_dir = os.getcwd()
    os.mkdir(f"{origin_dir}/run_{run_number}")
    os.chdir(f"{origin_dir}/run_{run_number}")
    with open(f"{variable_name}.pkl", 'wb') as file:
        pickle.dump(variable, file)
    os.chdir(origin_dir)
