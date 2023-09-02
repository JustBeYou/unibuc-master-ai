from pprint import pprint


def main():
    default_grammar = """
        S -> NP VP
        S -> VP
        NP -> DT NN
        NP -> DT JJ NN
        NP -> PRP
        VP -> VBP NP
        VP -> VBP VP
        VP -> VBG NP
        VP -> TO VP
        VP -> VB
        VP -> VB NP
        JJ -> "good"
        NN -> "show" | "book"
        PRP -> "I"
        VBP -> "am"
        VBG -> "watching"
        VB -> "show"
        DT -> "a" | "the"
        MD -> "will"
    """

    grammar = load_grammar(default_grammar)
    _, reverse_left_corner_table = build_left_corner_table(
        grammar)

    found, _ = parse("S", "I am watch a show".split(" "),
                     grammar, reverse_left_corner_table)
    print(f"found = {found}")

    found, solution = parse("S", "I am watching a show".split(" "),
                            grammar, reverse_left_corner_table)
    print(f"found = {found}, solution = {solution}")


steps = 0
debug = True


def parse(goal, words, grammar, rev_lc_table):
    global steps
    steps = 0

    bottom_up = [words[0]]
    top_down = [goal]
    words = words[:0:-1]
    found = parse_bt(bottom_up, top_down, words, grammar, rev_lc_table)
    return found, bottom_up


def parse_bt(bottom_up, top_down, words, grammar, rev_lc_table, depth=0):
    global steps

    if debug:
        print(f"(depth={depth}, step={steps})", bottom_up, top_down, words)
    steps += 1

    if len(bottom_up) == 0 or len(top_down) == 0:
        return False

    bottom_up_head = bottom_up[-1]
    top_down_head = top_down[-1]

    if bottom_up_head == top_down_head and len(words) == 0:
        if debug:
            print(f"(depth={depth}, step={steps}) FOUND")
        return True

    if bottom_up_head == top_down_head:
        words_head = words[-1]

        for production in grammar[top_down_head]['productions']:
            if production[0] != bottom_up[-2]:
                continue

            if len(production) == 1:
                continue

            top_down.append(production[1])
            bottom_up.append(words_head)
            words.pop()

            found = parse_bt(bottom_up, top_down, words,
                             grammar, rev_lc_table, depth+1)

            if found:
                return True

            top_down.pop()
            bottom_up.pop()
            words.append(words_head)

    if bottom_up_head in rev_lc_table:
        for left_corner in rev_lc_table[bottom_up_head]:
            bottom_up.append(left_corner)

            found = parse_bt(bottom_up, top_down, words,
                             grammar, rev_lc_table, depth+1)

            if found:
                return True

            bottom_up.pop()

    return False


def build_left_corner_table(grammar):
    table = {}

    for symbol in grammar:
        if symbol not in table:
            table[symbol] = set()

        for production in grammar[symbol]['productions']:
            table[symbol].add(production[0])

    reverse_table = {}
    for symbol in table:
        for production in table[symbol]:
            if production not in reverse_table:
                reverse_table[production] = set()

            reverse_table[production].add(symbol)

    return table, reverse_table


def load_grammar(grammar):
    grammar = [line.strip() for line in grammar.split("\n")]
    grammar = [line for line in grammar if line != ""]
    grammar = [[term.strip() for term in line.split(' ')] for line in grammar]
    grammar = [[term for term in line if term != ""] for line in grammar]

    parsed_grammar = {}

    for rule in grammar:
        assert len(rule) > 2
        assert rule[1] == "->"

        left_symbol = rule[0]
        if left_symbol not in parsed_grammar:
            parsed_grammar[left_symbol] = {"productions": [], "terminal": True}

        current_production = []
        for symbol in rule[2:]:
            if symbol == "|":
                parsed_grammar[left_symbol]["productions"].append(
                    current_production)
                current_production = []
            else:
                current_production.append(symbol.replace('"', ""))
                if symbol[0] != '"':
                    parsed_grammar[left_symbol]["terminal"] = False

        if len(current_production) > 0:
            parsed_grammar[left_symbol]["productions"].append(
                current_production)

    return parsed_grammar


if __name__ == "__main__":
    main()
