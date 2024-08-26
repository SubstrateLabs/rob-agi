from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_7e02026e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying and coloring specific 'L'-shaped or extended 'L'-shaped
    regions of black (0) squares to green (3). The algorithm follows these steps:
    1. Identify potential L-shapes in black regions
    2. Validate shapes to ensure they're surrounded by sky blue (8) or grid edges
    3. Score shapes based on size and position
    4. Transform the highest-scoring shape to green
    5. Repeat until no more valid shapes can be found

    Valid shapes must be completely surrounded by sky blue (8) squares or grid edges.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def is_valid_shape(shape: Set[Tuple[int, int]]) -> bool:
        for r, c in shape:
            for nr, nc in get_neighbors(r, c):
                if (nr, nc) not in shape and output_grid.get_cell(nr, nc) != 8 and (nr, nc) not in [(0, c), (r, 0), (rows-1, c), (r, cols-1)]:
                    return False
        return True

    def find_l_shapes() -> List[Set[Tuple[int, int]]]:
        shapes = []
        for r in range(rows):
            for c in range(cols):
                if output_grid.get_cell(r, c) == 0:
                    # Check for L-shape
                    l_shape = {(r, c), (r+1, c), (r, c+1)}
                    if all(0 <= x < rows and 0 <= y < cols and output_grid.get_cell(x, y) == 0 for x, y in l_shape):
                        if is_valid_shape(l_shape):
                            shapes.append(l_shape)
                    # Check for extended L-shape
                    ext_l_shape = {(r, c), (r+1, c), (r+2, c), (r, c+1), (r, c+2)}
                    if all(0 <= x < rows and 0 <= y < cols and output_grid.get_cell(x, y) == 0 for x, y in ext_l_shape):
                        if is_valid_shape(ext_l_shape):
                            shapes.append(ext_l_shape)
        return shapes

    def shape_score(shape: Set[Tuple[int, int]]) -> int:
        size = len(shape)
        # Prefer shapes closer to corners
        corner_distance = min(min(r, rows-1-r) + min(c, cols-1-c) for r, c in shape)
        return size * 10 - corner_distance

    while True:
        shapes = find_l_shapes()
        if not shapes:
            break

        shapes.sort(key=shape_score, reverse=True)
        best_shape = shapes[0]

        for r, c in best_shape:
            output_grid.set_cell(r, c, 3)

    return output_grid
