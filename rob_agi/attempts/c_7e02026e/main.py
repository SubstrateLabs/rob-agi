from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

from typing import List, Tuple, Set
import heapq

def solve_7e02026e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying and coloring specific 'L'-shaped or extended 'L'-shaped
    regions of black (0) squares to green (3). The algorithm follows these steps:
    1. Identify potential L-shapes in black regions
    2. Validate shapes to ensure they're surrounded by sky blue (8) or grid edges
    3. Score shapes based on size, position, and overall grid balance
    4. Transform the highest-scoring shape to green
    5. Re-evaluate remaining shapes and grid balance
    6. Repeat until a balanced state is achieved or no more valid shapes can be found

    The transformation aims to create a balanced distribution of green shapes while maintaining
    the overall aesthetic of the grid.
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

    def shape_score(shape: Set[Tuple[int, int]]) -> float:
        size = len(shape)
        corner_distance = min(min(r, rows-1-r) + min(c, cols-1-c) for r, c in shape)
        center_r, center_c = sum(r for r, _ in shape) / size, sum(c for _, c in shape) / size
        center_distance = abs(center_r - rows/2) + abs(center_c - cols/2)
        
        # Balance score: prefer shapes that improve overall balance
        green_count = sum(1 for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) == 3)
        balance_score = 1 / (1 + abs(green_count - 2))  # Aim for about 2 green shapes
        
        return size * 10 - corner_distance - center_distance * 0.5 + balance_score * 20

    def transform_shape(shape: Set[Tuple[int, int]]):
        for r, c in shape:
            output_grid.set_cell(r, c, 3)

    def is_balanced():
        green_count = sum(1 for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) == 3)
        return 2 <= green_count <= 3  # Allow 2 or 3 green shapes for balance

    shapes = find_l_shapes()
    shape_heap = [(-shape_score(shape), i, shape) for i, shape in enumerate(shapes)]
    heapq.heapify(shape_heap)

    while shape_heap and not is_balanced():
        _, _, best_shape = heapq.heappop(shape_heap)
        transform_shape(best_shape)
        
        # Re-evaluate remaining shapes
        shapes = find_l_shapes()
        shape_heap = [(-shape_score(shape), i, shape) for i, shape in enumerate(shapes)]
        heapq.heapify(shape_heap)

    return output_grid
