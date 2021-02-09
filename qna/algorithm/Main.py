import re

rList = ["^.*은", "^.*는", "^.*가", "^.*의", "^.*을", "^.*를"]
reList = []

for r in rList:
    rea = re.compile(r)
    reList.append(rea)


def getMain(str):
    for re in reList:
        if re.search(str):
            res = re.findall(str)

    res = res[0]
    res = res[:-1]

    return res
