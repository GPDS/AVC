def timeConversion(s):
    sep = s.split(":")
    firsttwo = int(sep[0])
    secondtwo = sep[1]
    thirdtwo = sep[2][:2]
    codification = sep[2][2:]
    rest = secondtwo + ":" + thirdtwo
    print(sep)
    restt = rest[0]+rest[1]
    firsttwo2 = 0
    if codification == "PM":
        if firsttwo < 12:
            firsttwo2 = str(firsttwo+12)
        else:
            firsttwo2 = str(firsttwo2)
    if codification == "AM":
        if firsttwo2 >=12:
            firsttwo2 = str(firsttwo-12)
        else:
            firsttwo2 = str(firsttwo2)
    stringf = firsttwo2+":" + rest
    return stringf;
def main():
 s = "9:02:05AM"
main()
