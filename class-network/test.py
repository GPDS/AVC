import os

items = os.listdir("val_path")
print(items)
val_files_normal = []
val_files_anormal = []
for item in items:
    
    for file in os.listdir(os.path.join("val_path",item)):
        arquivo = os.path.join("val_path",item,file)
        if item == "0":
            val_files_normal.append(arquivo)
        else:
            val_files_anormal.append(arquivo)

