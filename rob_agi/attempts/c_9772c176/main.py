from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List
import random

import random
from typing import List, Tuple, Set

def solve_9772c176(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a yellow (4) aura that responds to sky blue (8) shapes.
    
    The solution follows these steps:
    1. Identify sky blue shapes and create an initial yellow border around them.
    2. Extend yellow tendrils into black space, with a bias towards upward and sideways directions.
    3. Apply attractive forces from sky blue shapes to create a density gradient in the yellow aura.
    4. Ensure connectivity of all yellow pixels.
    5. Apply cellular automaton rules to refine the yellow aura.
    6. Make final adjustments for asymmetry and upward bias.

    Args:
    input_grid (ColoredGrid): The input grid containing sky blue shapes.

    Returns:
    ColoredGrid: The transformed grid with added yellow aura.
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

    def create_yellow_border(shapes: List[Set[Tuple[int, int]]]) -> Set[Tuple[int, int]]:
        border = set()
        for shape in shapes:
            for x, y in shape:
                for nx, ny in get_neighbors(x, y, diagonal=True):
                    if output_grid.get_cell(nx, ny) == 0:
                        output_grid.set_cell(nx, ny, 4)
                        border.add((nx, ny))
        return border

    def extend_yellow_tendrils(border: Set[Tuple[int, int]]):
        for x, y in border:
            length = random.randint(1, 5)
            cx, cy = x, y
            for _ in range(length):
                neighbors = [n for n in get_neighbors(cx, cy) if output_grid.get_cell(*n) == 0]
                if not neighbors:
                    break
                nx, ny = random.choice(neighbors)
                output_grid.set_cell(nx, ny, 4)
                cx, cy = nx, ny

    def connect_shapes(shapes: List[Set[Tuple[int, int]]]):
        for i, shape1 in enumerate(shapes):
            for shape2 in shapes[i+1:]:
                x1, y1 = random.choice(list(shape1))
                x2, y2 = random.choice(list(shape2))
                if abs(x1-x2) + abs(y1-y2) < max(rows, cols) // 4:
                    while (x1, y1) != (x2, y2):
                        if random.random() < 0.7:
                            x1 += 1 if x2 > x1 else -1 if x2 < x1 else 0
                        else:
                            y1 += 1 if y2 > y1 else -1 if y2 < y1 else 0
                        if output_grid.get_cell(x1, y1) == 0:
                            output_grid.set_cell(x1, y1, 4)

    def fill_large_black_spaces():
        for x in range(rows):
            for y in range(cols):
                if output_grid.get_cell(x, y) == 0:
                    if random.random() < 0.1:
                        output_grid.set_cell(x, y, 4)

    def ensure_connectivity():
        def flood_fill(start_x, start_y):
            stack = [(start_x, start_y)]
            visited = set()
            while stack:
                x, y = stack.pop()
                if (x, y) in visited:
                    continue
                visited.add((x, y))
                for nx, ny in get_neighbors(x, y):
                    if output_grid.get_cell(nx, ny) == 4:
                        stack.append((nx, ny))
            return visited

        yellow_pixels = [(x, y) for x in range(rows) for y in range(cols) if output_grid.get_cell(x, y) == 4]
        if not yellow_pixels:
            return

        connected = flood_fill(*yellow_pixels[0])
        for x, y in yellow_pixels:
            if (x, y) not in connected:
                path = []
                cx, cy = x, y
                while (cx, cy) not in connected:
                    path.append((cx, cy))
                    neighbors = get_neighbors(cx, cy)
                    cx, cy = min(neighbors, key=lambda n: min(abs(n[0]-tx) + abs(n[1]-ty) for tx, ty in connected))
                for px, py in path:
                    output_grid.set_cell(px, py, 4)

    def apply_cellular_automaton():
        nonlocal output_grid
        for _ in range(3):
            new_grid = output_grid.deep_copy()
            for x in range(rows):
                for y in range(cols):
                    if output_grid.get_cell(x, y) == 4:
                        neighbors = sum(1 for nx, ny in get_neighbors(x, y, diagonal=True) if output_grid.get_cell(nx, ny) == 4)
                        if neighbors < 2 or neighbors > 4:
                            new_grid.set_cell(x, y, 0)
                    elif output_grid.get_cell(x, y) == 0:
                        neighbors = sum(1 for nx, ny in get_neighbors(x, y, diagonal=True) if output_grid.get_cell(nx, ny) == 4)
                        if neighbors == 3:
                            new_grid.set_cell(x, y, 4)
            output_grid = new_grid

    def apply_density_gradient():
        for x in range(rows):
            for y in range(cols):
                if output_grid.get_cell(x, y) == 4:
                    distance = min(abs(x-bx) + abs(y-by) for shape in blue_shapes for bx, by in shape)
                    if random.random() < distance / (rows + cols):
                        output_grid.set_cell(x, y, 0)

    def enhance_asymmetry():
        for shape in blue_shapes:
            left = min(y for _, y in shape)
            right = max(y for _, y in shape)
            for x, y in shape:
                if y < (left + right) // 2:
                    for nx, ny in get_neighbors(x, y):
                        if output_grid.get_cell(nx, ny) == 0 and random.random() < 0.2:
                            output_grid.set_cell(nx, ny, 4)
                else:
                    for nx, ny in get_neighbors(x, y):
                        if output_grid.get_cell(nx, ny) == 4 and random.random() < 0.2:
                            output_grid.set_cell(nx, ny, 0)

    def reinforce_upward_bias():
        for shape in blue_shapes:
            top = min(x for x, _ in shape)
            bottom = max(x for x, _ in shape)
            mid = (top + bottom) // 2
            for x, y in [(x, y) for x, y in shape if x > mid]:
                if random.random() < 0.3:
                    for _ in range(random.randint(1, 3)):
                        if x > 0 and output_grid.get_cell(x-1, y) == 0:
                            output_grid.set_cell(x-1, y, 4)
                            x -= 1
                        else:
                            break

    # Main execution
    blue_shapes = find_blue_shapes()
    yellow_border = create_yellow_border(blue_shapes)
    extend_yellow_tendrils(yellow_border)
    connect_shapes(blue_shapes)
    fill_large_black_spaces()
    ensure_connectivity()
    apply_cellular_automaton()
    apply_density_gradient()
    enhance_asymmetry()
    reinforce_upward_bias()
    ensure_connectivity()  # Final connectivity check

    # Clear yellow pixels from the edge of the grid
    for x in range(rows):
        output_grid.set_cell(x, 0, 0)
        output_grid.set_cell(x, cols-1, 0)
    for y in range(cols):
        output_grid.set_cell(0, y, 0)
        output_grid.set_cell(rows-1, y, 0)

    return output_grid
