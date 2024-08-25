from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_20818e16(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extracting and rearranging colored shapes.
    
    1. Identify the background color and extract non-background shapes.
    2. Preserve the exact shape and size of each colored region.
    3. Arrange shapes in a compact manner, prioritizing larger shapes.
    4. Remove the background color and optimize the grid size.
    5. Ensure all original colors (except background) are represented in the output.
    
    Returns a new ColoredGrid with the transformed and compact arrangement.
    """
    background_color = identify_background_color(input_grid)
    shapes = extract_shapes(input_grid, background_color)
    arranged_grid = arrange_shapes(shapes)
    return ColoredGrid(values=arranged_grid)

def identify_background_color(grid: ColoredGrid) -> int:
    color_count = defaultdict(int)
    for row in grid.values:
        for color in row:
            color_count[color] += 1
    return max(color_count, key=color_count.get)

def extract_shapes(grid: ColoredGrid, background_color: int) -> List[Tuple[int, List[Tuple[int, int]]]]:
    shapes = []
    rows, cols = grid.get_dimensions()
    visited = set()

    def dfs(r, c, color):
        shape = []
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and 0 <= r < rows and 0 <= c < cols and grid.values[r][c] == color:
                visited.add((r, c))
                shape.append((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    stack.append((nr, nc))
        return shape

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != background_color:
                shape = dfs(r, c, grid.values[r][c])
                shapes.append((grid.values[r][c], shape))

    return sorted(shapes, key=lambda x: len(x[1]), reverse=True)

def arrange_shapes(shapes: List[Tuple[int, List[Tuple[int, int]]]]) -> List[List[int]]:
    total_area = sum(len(shape) for _, shape in shapes)
    grid_size = int(total_area ** 0.5) + 1
    grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

    def can_place(r, c, shape):
        return all(0 <= r + dr < grid_size and 0 <= c + dc < grid_size and grid[r + dr][c + dc] == 0
                   for dr, dc in shape)

    def place_shape(r, c, color, shape):
        for dr, dc in shape:
            grid[r + dr][c + dc] = color

    for color, shape in shapes:
        min_r = min(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        normalized_shape = [(r - min_r, c - min_c) for r, c in shape]

        placed = False
        for r in range(grid_size):
            for c in range(grid_size):
                if can_place(r, c, normalized_shape):
                    place_shape(r, c, color, normalized_shape)
                    placed = True
                    break
            if placed:
                break

    # Remove empty rows and columns
    grid = [row for row in grid if any(cell != 0 for cell in row)]
    grid = [list(col) for col in zip(*grid) if any(cell != 0 for cell in col)]

    return grid
