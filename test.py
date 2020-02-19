p = (1,-1)
for i in range(10):
    n = int(i/2) + 1
    for j in range(n):
        # (1,1) : print(p[0] + p[0] * j, p[1] + p[1] * (i - j))
        # (-1,1): print(p[0] + p[0] * (i - j), p[1] + p[1] * j)
        # (-1,-1):print(p[0] + p[0] * j, p[1] + p[1] * (i - j))
        # (1,-1): print(p[0] + p[0] * (i - j), p[1] + p[1] * j)
    print("")
