from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_7e02026e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying and coloring 'L'-shaped or extended 'L'-shaped
    regions of black (0) squares to green (3). The algorithm follows these steps:
    1. Identify potential L-shapes and extended L-shapes in black regions
    2. Validate shapes to ensure they're surrounded by sky blue (8) or grid edges
    3. Score shapes based on size, complexity, and contribution to overall pattern
    4. Transform highest-scoring shapes to green
    5. Repeat until no more valid shapes can be found or pattern is satisfactory

    Valid shapes must not connect directly to existing sky blue squares.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def is_valid_shape(shape: Set[Tuple[int, int]]) -> bool:
        for r, c in shape:
            for nr, nc in get_neighbors(r, c):
                if (nr, nc) not in shape and output_grid.get_cell(nr, nc) not in [0, 8] and (nr, nc) not in [(0, c), (r, 0), (rows-1, c), (r, cols-1)]:
                    return False
        return True

    def find_and_extend_l_shapes() -> List[Set[Tuple[int, int]]]:
        shapes = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if output_grid.get_cell(r, c) == 0 and (r, c) not in visited:
                    shape = set([(r, c)])
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        for nr, nc in get_neighbors(cr, cc):
                            if output_grid.get_cell(nr, nc) == 0 and (nr, nc) not in visited:
                                shape.add((nr, nc))
                                visited.add((nr, nc))
                                stack.append((nr, nc))
                    if len(shape) >= 3 and is_valid_shape(shape):
                        shapes.append(shape)
        return shapes

    def shape_score(shape: Set[Tuple[int, int]]) -> int:
        size = len(shape)
        complexity = sum(1 for r, c in shape if sum(1 for nr, nc in get_neighbors(r, c) if (nr, nc) in shape) > 1)
        return size * 10 + complexity * 5

    while True:
        shapes = find_and_extend_l_shapes()
        if not shapes:
            break

        shapes.sort(key=shape_score, reverse=True)
        best_shape = shapes[0]

        for r, c in best_shape:
            output_grid.set_cell(r, c, 3)

    return output_grid
