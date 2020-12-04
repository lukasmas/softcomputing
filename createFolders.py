#!/usr/bin/env python
import os
from glob import glob

for i in range(200):
    try:
        os.mkdir("./dataset2/" + "sub" + str(i + 1))
    except:
        print("exist")

os.chdir("./dataset2/")

folder = "./*.BMP"
bmps = glob(folder)

bmps.sort()
for b in bmps:
    toMove = b[2:len(b)]
    where = toMove.split("_")[0]
    print(where)
    try:
        os.rename("./" + toMove, "./sub" + where + "/" + toMove)
    except:
        print(toMove)
