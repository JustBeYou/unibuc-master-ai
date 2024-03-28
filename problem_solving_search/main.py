from argparse import ArgumentParser
from bfs import BreadthSearchFirst
from a_star import AStar
from npuzzle import NPuzzle, DirectionStr
from pytictoc import TicToc


def main():
    arg_parser = ArgumentParser(
        description="N-Puzzle solver using multiple search algorithms."
    )
    arg_parser.add_argument('N', type=int)
    args = arg_parser.parse_args()

    puzzle = NPuzzle(args.N)
    print("Puzzle:")
    print(puzzle)

    ### ### ### ### ###

    # print("A* (Naive)")
    # t = TicToc()
    # t.tic()
    # astar = AStar(puzzle)
    # solution = astar.search()
    # t.toc()

    # if args.N < 3:
    #     print("Solution: ", list(map(lambda x: DirectionStr[x], solution)))
    # print("Solution length:", len(solution))
    # print()

    ### ### ### ### ###

    print("A* (Manhattan)")
    t = TicToc()
    t.tic()
    astar = AStar(puzzle)
    solution = astar.search(cost_fn="manhattan")
    t.toc()

    if args.N < 3:
        print("Solution: ", list(map(lambda x: DirectionStr[x], solution)))
    print("Solution length:", len(solution))
    print()

    ### ### ### ### ###

    # print("BFS with depth and width limiting")
    # t = TicToc()
    # t.tic()
    # bfs = BreadthSearchFirst(puzzle)
    # solution = bfs.search()
    # t.toc()

    # if args.N < 3:
    #     print("Solution: ", list(map(lambda x: DirectionStr[x], solution)))
    # print("Solution length:", len(solution))
    # print()

    ### ### ### ### ###


if __name__ == "__main__":
    main()
