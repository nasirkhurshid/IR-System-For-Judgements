import os
folders = ["C.A.", "C.M.A.", "C.P.", "Const.P.", "Crl.A.", "Crl.P.", "S.M.C."]

for folder in folders:
    i = 0
    for filename in os.listdir(folder):
        path = './'+folder+'/'
        if filename.endswith(".pdf") or filename.endswith(".PDF"):
            i += 1
            oldPath = path + filename
            if (filename.startswith(folder.upper()[:-1]+'-No-')):
                continue
            newPath = path+folder.upper()[:-1]+'-No-'+str(i)+'____'+filename
            os.rename(oldPath, newPath)
