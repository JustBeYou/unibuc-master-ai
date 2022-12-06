from random import choice

vs=[f'c{i}' for i in range(20)]
vs=vs+[f'c{i}' for i in range(12)]
vs=vs+[f'c{i}' for i in range(8)]
vs=vs+[f'c{i}' for i in range(4)]
vs=vs+[f'c{i}' for i in range(4)]
vs=vs+[f'c{i}' for i in range(4)]
vs=vs+[f'c{i}' for i in range(4)]
vs=vs+[f'c{i}' for i in range(4)]
vs=vs+[f'c{i}' for i in range(4)]
vs=vs+[f'c{i}' for i in range(4)]
vs=vs+[f'c{i}' for i in range(2)]

for q in range(5):
    cnf = []

    for i in range(20000):
        x = []

        for j in range(3):
            neg = choice([1, 0])
            v = choice(vs)

            if neg: v = f"not({v})"

            x.append(v)
        cnf.append("[" + ', '.join(x) + "]")

    cnf = "[" + ', '.join(cnf) + "]"

    testcase = "test_case_KB("+cnf+").\n"

    open(f"testdata/performance{q}.txt", "w").write(testcase)
