#from haralick_features.regex_patterns import patterns
import pandas as pd
import regex as re
import numpy as np

pattern_a = r"^(?P<numero>[\d]*[\s]*)a(?P<numero2>[\s]*[\d]*)" ## Caso separado por "a" ex: 13 a 17
pattern_e = r"^(?P<numero>[\d]*[\s]*)e(?P<numero2>[\s]*[\d]*)"
pattern_blank = r"^(?P<numero>[\d]*)\s(?P<numero2>[\d]*)"


def separate_a(string: str):
    var = re.match(pattern_a,string)
    numero1 = var.group("numero")
    numero2 = var.group("numero2")
    list_values = [i for i in range(int(numero1),int(numero2)+1)]
    return list_values

def separate_e(string: str):
    var = re.match(pattern_e,string)
    numero1 = int(var.group("numero"))
    numero2 = int(var.group("numero2"))
    return [numero1,numero2]

def separate_blank(string: str):
    var = re.match(pattern_blank,string)
    numero1 = int(var.group("numero"))
    numero2 = int(var.group("numero2"))
    return [numero1,numero2]

def separe_corte(corte: list) -> list:
    corte_valores = []
    for i in corte:
        i = i.strip()
        if re.fullmatch(pattern_a,i):
            corte_valores += separate_a(i)
        if re.fullmatch(pattern_e,i):
            corte_valores += separate_e(i)
        try:
            corte_valores += int(i)
        except:
            pass
        if re.fullmatch(pattern_blank,i):
            corte_valores += separate_blank(i)
    return sorted(corte_valores)
        
def get_avci(path: str) -> list:
    db = pd.read_csv(path)
    db["AVCi"].fillna(0, inplace= True)
    db_temp = db.loc[:,["Renomeação","AVCi"]]
    return db_temp       


def treat_after_merge(db):
    db["AVCi"] = db.apply(lambda x:x["AVCi_x"] if x["AVCi_x"] == '1' else x["AVCi_y"], axis = 1)
    db["AVCi"] = db.apply(lambda x:1 if x["AVCi"] == '1' else x["AVCi"], axis =1)
    db.drop(columns = ["AVCi_x","AVCi_y"], inplace = True)
    return db
        

def separete_cortes(cortes: list) -> list:
    list_string = []
    cortes_valores = []
    for i in cortes:
        if "/" in i:
            list_string = i.split('/')
        elif "," in i:
            list_string = i.split(',')
        else:
            list_string = [i]
        if len(list_string) >1:
            print("mydebug")            
        corte_valores = separe_corte(list_string)
        cortes_valores.append(corte_valores)
    return cortes_valores



pattern = r"^Unnamed:\s[\d]*"
db = pd.read_csv("BASE1.csv")
db_avci = get_avci("planilha_avc_csv.csv")
db = pd.merge(db,db_avci, on = "Renomeação")
db = treat_after_merge(db)
print(db.head(10))
for col in db.columns:
    if re.match(pattern, col):
        print(col)
        db.drop(columns = [col], inplace= True)
db["cortes"].fillna("0", inplace= True)
db_avci = db.loc[(db["AVCi"] == 1.0)]
print(db_avci)
cortes = db_avci.cortes.tolist()
ident = db_avci.Renomeação.tolist()
cortes_tratados = separete_cortes(cortes)
db_avci["cortes_tratados"] = cortes_tratados
db_temp = db_avci.loc[:,["Renomeação","cortes_tratados"]]
m = pd.merge(db, db_temp, how = 'left', on = "Renomeação")
print(m.shape)
m.to_csv("avc_controle_filtrado.csv")
