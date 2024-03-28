from npuzzle import NPuzzle, Directions


class BreadthSearchFirst:
    def __init__(self, state: NPuzzle, width_limit=50_000, discard_factor=2, depth_limit=15_000):
        self.state = state
        self.width_limit = width_limit
        self.discard_factor = discard_factor
        self.depth_limit = depth_limit

    def search(self):
        solved = False
        queue = [(self.state, [])]
        visited = set()
        nodes = 0

        while not solved and len(queue) > 0:
            state, moves = queue.pop()

            h = state.hash()
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

                if len(new_moves) > self.depth_limit:
                    continue

                nodes += 1

                queue.append((new_state, new_moves))

                if len(queue) > self.width_limit:
                    queue = queue[:int(len(queue)/self.discard_factor)]

        raise RuntimeError("Could not solve puzzle!")
