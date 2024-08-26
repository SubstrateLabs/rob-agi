from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List, Set
from collections import deque
import random

def solve_9772c176(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a yellow (4) shadow to sky blue (8) shapes.
    
    The solution follows these steps:
    1. Identify sky blue shapes using flood fill.
    2. For each sky blue shape:
       a. Create an expanded bounding box.
       b. Generate a yellow shadow based on the sky blue shape, with focus on bottom and right sides.
       c. Add irregularities and randomness to the shadow.
    3. Create disconnected shadow elements (small dots and tendrils).
    4. Refine shadows by removing isolated pixels and ensuring no direct contact with sky blue.
    5. Add final touch-ups and verify the result.

    Args:
    input_grid (ColoredGrid): The input grid containing sky blue shapes.

    Returns:
    ColoredGrid: The transformed grid with added yellow shadows.
    """
    random.seed(42)  # Set seed for reproducibility
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
                    queue = deque([(x, y)])
                    while queue:
                        cx, cy = queue.popleft()
                        if (cx, cy) in visited:
                            continue
                        visited.add((cx, cy))
                        shape.add((cx, cy))
                        for nx, ny in get_neighbors(cx, cy):
                            if output_grid.get_cell(nx, ny) == 8:
                                queue.append((nx, ny))
                    shapes.append(shape)
        return shapes

    def get_bounding_box(shape: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        x_coords, y_coords = zip(*shape)
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def create_expanded_box(box: Tuple[int, int, int, int], expansion: int) -> Tuple[int, int, int, int]:
        top, left, bottom, right = box
        return (max(0, top - expansion), 
                max(0, left - expansion), 
                min(rows - 1, bottom + expansion), 
                min(cols - 1, right + expansion))

    def generate_shadow(shape: Set[Tuple[int, int]], box: Tuple[int, int, int, int]) -> Set[Tuple[int, int]]:
        top, left, bottom, right = box
        shadow = set()
        for x in range(top, bottom + 1):
            for y in range(left, right + 1):
                if (x, y) not in shape and any((nx, ny) in shape for nx, ny in get_neighbors(x, y, diagonal=True)):
                    # Higher probability for bottom and right sides
                    if x > (top + bottom) // 2 or y > (left + right) // 2:
                        if random.random() < 0.9:
                            shadow.add((x, y))
                    else:
                        if random.random() < 0.5:
                            shadow.add((x, y))
        return shadow

    def apply_shadow(shadow: Set[Tuple[int, int]]):
        for x, y in shadow:
            if all(output_grid.get_cell(nx, ny) != 8 for nx, ny in get_neighbors(x, y)):
                output_grid.set_cell(x, y, 4)

    def add_irregularities(shadow: Set[Tuple[int, int]]):
        new_shadow = shadow.copy()
        for x, y in shadow:
            if random.random() < 0.3:
                for nx, ny in get_neighbors(x, y):
                    if (nx, ny) not in shadow and output_grid.get_cell(nx, ny) == 0:
                        new_shadow.add((nx, ny))
        return new_shadow

    def add_disconnected_elements(shape: Set[Tuple[int, int]], box: Tuple[int, int, int, int]):
        top, left, bottom, right = box
        for _ in range((bottom - top + right - left) // 3):
            x = random.randint(top, bottom)
            y = random.randint(left, right)
            if (x, y) not in shape and output_grid.get_cell(x, y) == 0:
                if all(output_grid.get_cell(nx, ny) != 8 for nx, ny in get_neighbors(x, y)):
                    output_grid.set_cell(x, y, 4)

    def refine_shadow():
        for x in range(rows):
            for y in range(cols):
                if output_grid.get_cell(x, y) == 4:
                    yellow_neighbors = sum(1 for nx, ny in get_neighbors(x, y) if output_grid.get_cell(nx, ny) == 4)
                    if yellow_neighbors == 0:
                        output_grid.set_cell(x, y, 0)

    def add_final_touches():
        for x in range(rows):
            for y in range(cols):
                if output_grid.get_cell(x, y) == 0:
                    if any(output_grid.get_cell(nx, ny) == 4 for nx, ny in get_neighbors(x, y, diagonal=True)):
                        if random.random() < 0.2:
                            output_grid.set_cell(x, y, 4)

    blue_shapes = find_blue_shapes()
    expansion = 4  # Increased expanded box size

    for shape in blue_shapes:
        box = get_bounding_box(shape)
        expanded_box = create_expanded_box(box, expansion)
        shadow = generate_shadow(shape, expanded_box)
        shadow = add_irregularities(shadow)
        apply_shadow(shadow)
        add_disconnected_elements(shape, expanded_box)

    refine_shadow()
    add_final_touches()

    # Ensure no yellow pixels touch blue pixels
    for x in range(rows):
        for y in range(cols):
            if output_grid.get_cell(x, y) == 4 and any(output_grid.get_cell(nx, ny) == 8 for nx, ny in get_neighbors(x, y)):
                output_grid.set_cell(x, y, 0)

    return output_grid
