from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_7e02026e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying and coloring 'L'-shaped or extended 'L'-shaped
    regions of black (0) squares to green (3). The algorithm follows these steps:
    1. Identify contiguous black regions
    2. For each region, find valid 'L' or extended 'L' shapes
    3. Select the largest and most complex shape
    4. Color the selected shape green
    5. Repeat until no more valid shapes can be found

    Valid shapes must be surrounded by sky blue (8) or grid edges and not connect
    directly to existing sky blue squares.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def find_black_regions() -> List[Set[Tuple[int, int]]]:
        visited = set()
        regions = []
        for r in range(rows):
            for c in range(cols):
                if output_grid.get_cell(r, c) == 0 and (r, c) not in visited:
                    region = set()
                    stack = [(r, c)]
                    while stack:
                        curr_r, curr_c = stack.pop()
                        if (curr_r, curr_c) not in visited and output_grid.get_cell(curr_r, curr_c) == 0:
                            visited.add((curr_r, curr_c))
                            region.add((curr_r, curr_c))
                            stack.extend(get_neighbors(curr_r, curr_c))
                    regions.append(region)
        return regions

    def is_valid_shape(shape: Set[Tuple[int, int]]) -> bool:
        for r, c in shape:
            for nr, nc in get_neighbors(r, c):
                if (nr, nc) not in shape and output_grid.get_cell(nr, nc) != 8 and (nr, nc) not in [(0, c), (r, 0), (rows-1, c), (r, cols-1)]:
                    return False
        return True

    def find_l_shapes(region: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
        l_shapes = []
        for r, c in region:
            for dr1, dc1 in [(0,1), (1,0), (0,-1), (-1,0)]:
                for dr2, dc2 in [(0,1), (1,0), (0,-1), (-1,0)]:
                    if dr1 != dr2 and dc1 != dc2:
                        shape = {(r, c), (r+dr1, c+dc1), (r+dr2, c+dc2)}
                        if shape.issubset(region) and is_valid_shape(shape):
                            l_shapes.append(shape)
        return l_shapes

    def extend_shape(shape: Set[Tuple[int, int]], region: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        extended = shape.copy()
        frontier = shape.copy()
        while frontier:
            r, c = frontier.pop()
            for nr, nc in get_neighbors(r, c):
                if (nr, nc) in region and (nr, nc) not in extended:
                    new_shape = extended | {(nr, nc)}
                    if is_valid_shape(new_shape):
                        extended.add((nr, nc))
                        frontier.add((nr, nc))
        return extended

    def shape_score(shape: Set[Tuple[int, int]]) -> int:
        return len(shape) * 10 + sum(1 for r, c in shape if any((nr, nc) in shape for nr, nc in get_neighbors(r, c)))

    while True:
        regions = find_black_regions()
        if not regions:
            break

        best_shape = None
        best_score = -1

        for region in regions:
            l_shapes = find_l_shapes(region)
            for shape in l_shapes:
                extended = extend_shape(shape, region)
                score = shape_score(extended)
                if score > best_score:
                    best_score = score
                    best_shape = extended

        if best_shape:
            for r, c in best_shape:
                output_grid.set_cell(r, c, 3)
        else:
            break

    return output_grid
