from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict
import time

def solve_20818e16(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extracting and rearranging colored shapes.
    
    1. Identify the background color and extract non-background shapes.
    2. Preserve the exact shape and size of each colored region.
    3. Arrange shapes in a compact manner, allowing rotations.
    4. Remove the background color and optimize the grid size.
    5. Ensure all original colors (except background) are represented in the output.
    
    Returns a new ColoredGrid with the transformed and compact arrangement.
    """
    background_color = identify_background_color(input_grid)
    shapes = extract_shapes(input_grid, background_color)
    arranged_grid = find_optimal_arrangement(shapes)
    return ColoredGrid(values=arranged_grid)

def identify_background_color(grid: ColoredGrid) -> int:
    color_count = defaultdict(int)
    for row in grid.values:
        for color in row:
            color_count[color] += 1
    return max(color_count, key=color_count.get)

def extract_shapes(grid: ColoredGrid, background_color: int) -> List[Tuple[int, List[List[int]]]]:
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
                shapes.append((grid.values[r][c], shape_grid))

    return sorted(shapes, key=lambda x: len(x[1]) * len(x[1][0]), reverse=True)

def rotate_shape(shape: List[List[int]]) -> List[List[int]]:
    return [list(row) for row in zip(*shape[::-1])]

def find_optimal_arrangement(shapes: List[Tuple[int, List[List[int]]]]) -> List[List[int]]:
    total_area = sum(len(shape) * len(shape[0]) for _, shape in shapes)
    min_size = int(total_area ** 0.5)
    max_size = sum(max(len(shape), len(shape[0])) for _, shape in shapes)

    def can_place(grid, shape, r, c):
        for i in range(len(shape)):
            for j in range(len(shape[0])):
                if shape[i][j] != 0:
                    if r + i >= len(grid) or c + j >= len(grid[0]) or grid[r + i][c + j] != 0:
                        return False
        return True

    def place_shape(grid, shape, r, c):
        for i in range(len(shape)):
            for j in range(len(shape[0])):
                if shape[i][j] != 0:
                    grid[r + i][c + j] = shape[i][j]

    def remove_shape(grid, shape, r, c):
        for i in range(len(shape)):
            for j in range(len(shape[0])):
                if shape[i][j] != 0:
                    grid[r + i][c + j] = 0

    def arrange_shapes(grid, shapes, index=0):
        if index == len(shapes):
            return True

        color, shape = shapes[index]
        for rotation in range(4):
            for r in range(len(grid)):
                for c in range(len(grid[0])):
                    if can_place(grid, shape, r, c):
                        place_shape(grid, shape, r, c)
                        if arrange_shapes(grid, shapes, index + 1):
                            return True
                        remove_shape(grid, shape, r, c)
            shape = rotate_shape(shape)
        return False

    start_time = time.time()
    for size in range(min_size, max_size + 1):
        grid = [[0 for _ in range(size)] for _ in range(size)]
        if arrange_shapes(grid, shapes):
            # Trim empty rows and columns
            grid = [row for row in grid if any(cell != 0 for cell in row)]
            grid = [list(col) for col in zip(*grid) if any(cell != 0 for cell in col)]
            return grid
        if time.time() - start_time > 5:  # 5 seconds timeout
            break

    # If no solution found, return the best effort arrangement
    return grid
