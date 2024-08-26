from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List
import random

import random
from typing import List, Tuple, Set

def solve_9772c176(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a yellow (4) shadow to sky blue (8) shapes.
    
    The solution follows these steps:
    1. Identify sky blue shapes using flood fill.
    2. For each sky blue shape:
       a. Create an offset bounding box.
       b. Generate a simplified yellow shape based on the sky blue shape.
       c. Apply the yellow shape to the grid, ensuring a gap from sky blue pixels.
    3. Refine yellow shapes by smoothing edges and removing edge-adjacent pixels.
    4. Ensure all original sky blue pixels are unchanged.

    Args:
    input_grid (ColoredGrid): The input grid containing sky blue shapes.

    Returns:
    ColoredGrid: The transformed grid with added yellow shadows.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def get_neighbors(x: int, y: int, diagonal: bool = False) -> List[Tuple[int, int]]:
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        if diagonal:
            directions += [(-1,-1), (-1,1), (1,-1), (1,1)]
        return [(x+dx, y+dy) for dx, dy in directions if 0 <= x+dx < rows and 0 <= y+dy < cols]

    def find_blue_shapes() -> List[Set[Tuple[int, int]]]:
        shapes = []
        visited = set()
        for x in range(rows):
            for y in range(cols):
                if output_grid.get_cell(x, y) == 8 and (x, y) not in visited:
                    shape = set()
                    stack = [(x, y)]
                    while stack:
                        cx, cy = stack.pop()
                        if (cx, cy) in visited:
                            continue
                        visited.add((cx, cy))
                        shape.add((cx, cy))
                        for nx, ny in get_neighbors(cx, cy):
                            if output_grid.get_cell(nx, ny) == 8:
                                stack.append((nx, ny))
                    shapes.append(shape)
        return shapes

    def get_bounding_box(shape: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        x_coords, y_coords = zip(*shape)
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def create_offset_box(box: Tuple[int, int, int, int], offset: int) -> Tuple[int, int, int, int]:
        top, left, bottom, right = box
        return (top + offset, left + offset, bottom + offset, right + offset)

    def simplify_shape(shape: Set[Tuple[int, int]], box: Tuple[int, int, int, int]) -> Set[Tuple[int, int]]:
        top, left, bottom, right = box
        simplified = set()
        for x in range(top, bottom + 1):
            for y in range(left, right + 1):
                neighbors = sum(1 for nx, ny in get_neighbors(x, y) if (nx, ny) in shape)
                if neighbors >= 2:
                    simplified.add((x, y))
        return simplified

    def apply_yellow_shape(shape: Set[Tuple[int, int]], box: Tuple[int, int, int, int]):
        top, left, bottom, right = box
        for x in range(max(0, top), min(rows, bottom + 1)):
            for y in range(max(0, left), min(cols, right + 1)):
                if (x, y) in shape and all(output_grid.get_cell(nx, ny) != 8 for nx, ny in get_neighbors(x, y)):
                    output_grid.set_cell(x, y, 4)

    def smooth_edges():
        for x in range(rows):
            for y in range(cols):
                if output_grid.get_cell(x, y) == 4:
                    yellow_neighbors = sum(1 for nx, ny in get_neighbors(x, y) if output_grid.get_cell(nx, ny) == 4)
                    if yellow_neighbors <= 1:
                        output_grid.set_cell(x, y, 0)

    blue_shapes = find_blue_shapes()
    offset = 2  # Fixed offset

    for shape in blue_shapes:
        box = get_bounding_box(shape)
        offset_box = create_offset_box(box, offset)
        yellow_shape = simplify_shape(shape, offset_box)
        apply_yellow_shape(yellow_shape, offset_box)

    smooth_edges()

    # Remove yellow pixels from the edges
    for x in range(rows):
        if output_grid.get_cell(x, 0) == 4:
            output_grid.set_cell(x, 0, 0)
        if output_grid.get_cell(x, cols-1) == 4:
            output_grid.set_cell(x, cols-1, 0)
    for y in range(cols):
        if output_grid.get_cell(0, y) == 4:
            output_grid.set_cell(0, y, 0)
        if output_grid.get_cell(rows-1, y) == 4:
            output_grid.set_cell(rows-1, y, 0)

    return output_grid
