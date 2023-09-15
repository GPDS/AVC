import pandas as pd
import numpy as np
import re

def fix_sex():
    regex_pattern = r"^(?P<age>\d*)\w"
    path = "results_haralick_tomo.csv"
    db = pd.read_csv(path)
    list_sex = db.patient_sex.to_list()
    list_sex_enumerated = []
    list_age = db.patient_age.to_list()
    list_age_fixed = []
    count = 0
    for i in list_sex:
        match = re.match(regex_pattern,list_age[0])
        age = match.group("age")
        list_age_fixed.append(age)
        if i == "F":
            list_sex_enumerated.append(0)
        elif i == "M":
            list_sex_enumerated.append(1)
        else:
            list_sex_enumerated.append(3)
        count+=1


    db["patient_sex_usefull"] = list_sex_enumerated
    db["patient_age_usefull"] = list_age_fixed
    db = db.drop(["Unnamed: 0"],axis = 1)
    db.to_csv("haralick_tomo_fixed.csv") 




if __name__ == "__main__":
    fix_sex()
