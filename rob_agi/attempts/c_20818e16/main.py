from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict
import time

from typing import List, Tuple, NamedTuple
from collections import defaultdict

class Shape(NamedTuple):
    color: int
    height: int
    width: int
    structure: List[List[int]]
    original_position: Tuple[int, int]

def solve_20818e16(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extracting and rearranging colored shapes.
    
    1. Identify the background color and extract non-background shapes.
    2. Preserve the exact shape and size of each colored region.
    3. Sort shapes by area (ascending), then by color, then by original position.
    4. Arrange shapes in a compact manner, starting from the smallest possible grid.
    5. Optimize the final arrangement by removing empty space.
    6. Ensure all original colors (except background) are represented in the output.
    
    The algorithm uses a deterministic approach to find a compact arrangement,
    trying different grid sizes and positions for each shape. It starts with a minimum
    grid size and expands if necessary. The final arrangement is optimized by
    removing empty rows and columns.
    
    Returns a new ColoredGrid with the transformed and compact arrangement.
    """
    background_color = identify_background_color(input_grid)
    shapes = extract_shapes(input_grid, background_color)
    sorted_shapes = sort_shapes(shapes)
    arranged_grid = find_optimal_arrangement(sorted_shapes)
    optimized_grid = optimize_grid(arranged_grid)
    return ColoredGrid(values=optimized_grid)

def identify_background_color(grid: ColoredGrid) -> int:
    color_count = defaultdict(int)
    for row in grid.values:
        for color in row:
            color_count[color] += 1
    return max(color_count, key=color_count.get)

def extract_shapes(grid: ColoredGrid, background_color: int) -> List[Shape]:
    shapes = []
    rows, cols = grid.get_dimensions()
    visited = set()

    def dfs(r, c, color):
        shape = []
        min_r, min_c = r, c
        max_r, max_c = r, c
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and 0 <= r < rows and 0 <= c < cols and grid.values[r][c] == color:
                visited.add((r, c))
                shape.append((r, c))
                min_r, min_c = min(min_r, r), min(min_c, c)
                max_r, max_c = max(max_r, r), max(max_c, c)
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    stack.append((nr, nc))
        return shape, (min_r, min_c, max_r, max_c)

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != background_color:
                shape, (min_r, min_c, max_r, max_c) = dfs(r, c, grid.values[r][c])
                shape_grid = [[0 for _ in range(max_c - min_c + 1)] for _ in range(max_r - min_r + 1)]
                for sr, sc in shape:
                    shape_grid[sr - min_r][sc - min_c] = grid.values[r][c]
                shapes.append(Shape(grid.values[r][c], max_r - min_r + 1, max_c - min_c + 1, shape_grid, (min_r, min_c)))

    return shapes

def sort_shapes(shapes: List[Shape]) -> List[Shape]:
    return sorted(shapes, key=lambda s: (s.height * s.width, s.color, s.original_position))

def find_optimal_arrangement(shapes: List[Shape]) -> List[List[int]]:
    total_area = sum(s.height * s.width for s in shapes)
    min_size = max(int(total_area ** 0.5), max(max(s.height, s.width) for s in shapes))

    def can_place(grid, shape, r, c):
        if r + shape.height > len(grid) or c + shape.width > len(grid[0]):
            return False
        return all(grid[r+i][c+j] == 0 or shape.structure[i][j] == 0
                   for i in range(shape.height) for j in range(shape.width))

    def place_shape(grid, shape, r, c):
        for i in range(shape.height):
            for j in range(shape.width):
                if shape.structure[i][j] != 0:
                    grid[r+i][c+j] = shape.structure[i][j]

    while True:
        grid = [[0 for _ in range(min_size)] for _ in range(min_size)]
        for shape in shapes:
            placed = False
            for r in range(min_size):
                for c in range(min_size):
                    if can_place(grid, shape, r, c):
                        place_shape(grid, shape, r, c)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                min_size += 1
                break
        else:
            return grid

def optimize_grid(grid: List[List[int]]) -> List[List[int]]:
    # Remove empty rows and columns
    rows = [i for i, row in enumerate(grid) if any(cell != 0 for cell in row)]
    cols = [j for j in range(len(grid[0])) if any(row[j] != 0 for row in grid)]
    
    return [[grid[i][j] for j in cols] for i in rows]
