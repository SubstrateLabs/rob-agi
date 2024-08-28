from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List, Set
from collections import deque
import random
import math

def solve_9772c176(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a yellow (4) shadow to sky blue (8) shapes.
    
    The solution follows these steps:
    1. Identify sky blue shapes using flood fill.
    2. Create a shadow intensity map based on distance from sky blue shapes.
    3. Generate shadows with focus on bottom and right sides of shapes.
    4. Create bridges between shapes if multiple exist.
    5. Add scattered shadow pixels in black space.
    6. Refine shadow edges and ensure no direct contact with sky blue.
    7. Convert shadow intensity map to yellow pixels.
    8. Final cleanup to remove isolated pixels and ensure shadow continuity.

    Args:
    input_grid (ColoredGrid): The input grid containing sky blue shapes.

    Returns:
    ColoredGrid: The transformed grid with added yellow shadows.
    """
    random.seed(42)  # Set seed for reproducibility
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    shadow_map = [[0.0 for _ in range(cols)] for _ in range(rows)]

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

    def create_shadow_map(shapes: List[Set[Tuple[int, int]]]):
        for x in range(rows):
            for y in range(cols):
                if output_grid.get_cell(x, y) != 8:
                    min_distance = min(min(math.sqrt((x-bx)**2 + (y-by)**2) for bx, by in shape) for shape in shapes)
                    intensity = max(0, 1 - min_distance / 10)  # Adjust the divisor to control shadow spread
                    shadow_map[x][y] = intensity

    def apply_directional_bias():
        for x in range(rows):
            for y in range(cols):
                if shadow_map[x][y] > 0:
                    # Increase intensity towards bottom-right
                    shadow_map[x][y] *= 1 + (x / rows + y / cols) / 2

    def create_bridges(shapes: List[Set[Tuple[int, int]]]):
        if len(shapes) < 2:
            return
        for i, shape1 in enumerate(shapes):
            for shape2 in shapes[i+1:]:
                start = min(shape1, key=lambda p: min(math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2) for q in shape2))
                end = min(shape2, key=lambda p: math.sqrt((p[0]-start[0])**2 + (p[1]-start[1])**2))
                x, y = start
                while (x, y) != end:
                    dx = 1 if end[0] > x else -1 if end[0] < x else 0
                    dy = 1 if end[1] > y else -1 if end[1] < y else 0
                    x += dx
                    y += dy
                    if 0 <= x < rows and 0 <= y < cols and output_grid.get_cell(x, y) == 0:
                        shadow_map[x][y] = max(shadow_map[x][y], 0.5)  # Adjust intensity as needed

    def add_scattered_pixels():
        for x in range(rows):
            for y in range(cols):
                if output_grid.get_cell(x, y) == 0 and shadow_map[x][y] == 0:
                    if random.random() < 0.05 * (x / rows + y / cols):  # Higher probability towards bottom-right
                        shadow_map[x][y] = 0.3  # Adjust intensity as needed

    def refine_edges():
        for x in range(rows):
            for y in range(cols):
                if shadow_map[x][y] > 0:
                    neighbors = get_neighbors(x, y, diagonal=True)
                    avg_intensity = sum(shadow_map[nx][ny] for nx, ny in neighbors) / len(neighbors)
                    shadow_map[x][y] = (shadow_map[x][y] + avg_intensity) / 2

    def convert_to_yellow():
        for x in range(rows):
            for y in range(cols):
                if shadow_map[x][y] > 0.3 and output_grid.get_cell(x, y) == 0:  # Adjust threshold as needed
                    if all(output_grid.get_cell(nx, ny) != 8 for nx, ny in get_neighbors(x, y)):
                        output_grid.set_cell(x, y, 4)

    def cleanup():
        for x in range(rows):
            for y in range(cols):
                if output_grid.get_cell(x, y) == 4:
                    neighbors = sum(1 for nx, ny in get_neighbors(x, y, diagonal=True) if output_grid.get_cell(nx, ny) == 4)
                    if neighbors == 0:
                        output_grid.set_cell(x, y, 0)

    blue_shapes = find_blue_shapes()
    create_shadow_map(blue_shapes)
    apply_directional_bias()
    create_bridges(blue_shapes)
    add_scattered_pixels()
    refine_edges()
    convert_to_yellow()
    cleanup()

    return output_grid
