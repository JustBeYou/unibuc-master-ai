from queue import PriorityQueue

from npuzzle import Directions


class AStar:
    def __init__(self, state):
        self.state = state

    def search(self, cost_fn=None):
        if cost_fn is None or cost_fn == 'naive':
            cost_fn = self._cost_naive
        elif cost_fn == 'manhattan':
            cost_fn = self._cost_manhattan
        else:
            raise RuntimeError("Unkown cost function")

        queue = PriorityQueue()
        visited = set()
        nodes = 0

        queue.put((cost_fn(self.state), nodes, self.state, []))

        while not queue.empty():
            _, _, state, moves = queue.get()

            h = str(state)
            if h in visited:
                continue

            visited.add(h)

            for direction in Directions:
                if not state.can_move(direction):
                    continue

                new_state = state.clone()
                new_state.move(direction)
                new_moves = moves[:] + [direction]

                if new_state.is_solved():
                    print("[DEBUG]", {"nodes": nodes, "visited": len(visited)})
                    return new_moves

                if new_state.hash() in visited:
                    continue

                nodes += 1

                cost = cost_fn(new_state)
                queue.put((cost, nodes, new_state, new_moves))

        raise RuntimeError("Could not solve puzzle!")

    def _cost_naive(self, state):
        c = 0

        for i in range(state.n):
            for j in range(state.n):
                if i*state.n+j+1 != state.board[i][j]:
                    c += 1

        return c

    def _cost_manhattan(self, state):
        c = 0

        for i in range(state.n):
            for j in range(state.n):
                d = abs(i*state.n+j+1 - state.board[i][j])
                c += d // state.n + d % state.n

        return c
