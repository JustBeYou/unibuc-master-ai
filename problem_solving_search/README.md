BFS vs A* on solving N-Puzzle
===

## BFS

* Affected by state explosion and requires a high amount of memory
* Finds the best series of moves
* Can be quite slow
* Works in a reasonable amount of time only for N <= 3

Examples:

```
Puzzle:
 3 1
 4 2

BFS with depth and width limiting
[DEBUG] {'nodes': 9, 'visited': 9}
Elapsed time is 0.000283 seconds.
Solution:  ['RIGHT', 'UP', 'LEFT', 'DOWN', 'RIGHT', 'UP', 'LEFT', 'DOWN', 'RIGHT']
Solution length: 9
```

```
Puzzle:
 8 4 2
 6 9 7
 1 5 3

BFS with depth and width limiting
[DEBUG] {'nodes': 153549, 'visited': 142066}
Elapsed time is 26.057129 seconds.
Solution length: 15000
```

## A*

* Uinsg a heurisitc drastically reduces the amount of explored nodes
* Works well up to N = 4 with really dumb heuristic
* For larger N it requires a better heuristic/other optimizations like a pattern database

Examples:
```
Puzzle:
 3 1
 4 2
A*
[DEBUG] {'nodes': 4, 'visited': 3}
Elapsed time is 0.000268 seconds.
Solution:  ['UP', 'RIGHT', 'DOWN']
Solution length: 3
```

```
Puzzle:
 8 4 2
 6 9 7
 1 5 3
A*
[DEBUG] {'nodes': 317, 'visited': 115}
Elapsed time is 0.009051 seconds.
Solution length: 44
```

```
Puzzle:
 15  5  9  1
  4 10 14 16
  6 13  2 11
 12  3  7  8
A*
[DEBUG] {'nodes': 22016, 'visited': 7217}
Elapsed time is 0.629713 seconds.
Solution length: 192
```

```
Puzzle:
 10 13  8 16
  9  1  5  4
 14  2  6  3
 15 12  7 11
A* (Naive)
[DEBUG] {'nodes': 61750, 'visited': 19681}
Elapsed time is 1.844162 seconds.
Solution length: 201

A* (Manhattan)
[DEBUG] {'nodes': 18519, 'visited': 6171}
Elapsed time is 0.568135 seconds.
Solution length: 213
```