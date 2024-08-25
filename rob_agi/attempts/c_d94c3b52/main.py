from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_d94c3b52(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following steps:
    1. Analyzes the input grid to identify non-black shapes and their positions.
    2. Creates a pattern template dividing the grid into alternating zones.
    3. Applies color transformations based on the template and existing sky blue shapes.
    4. Balances color distribution and maintains structure integrity.
    5. Ensures contrast between adjacent shapes and consistency across similar patterns.
    6. Handles edge cases and makes final adjustments for visual appeal.
    """
    pattern_template = create_pattern_template(input_grid)
    new_grid = apply_transformations(input_grid, pattern_template)
    new_grid = balance_colors(new_grid)
    new_grid = ensure_contrast(new_grid)
    return new_grid

def create_pattern_template(grid: ColoredGrid) -> List[List[int]]:
    rows, cols = grid.get_dimensions()
    template = [[0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8:  # Sky blue
                template[r][c] = 2  # Fixed zone
            else:
                template[r][c] = (r + c) % 2  # Alternating zones
    return template

def apply_transformations(grid: ColoredGrid, template: List[List[int]]) -> ColoredGrid:
    new_grid = grid.deep_copy()
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0:  # Non-black cell
                if grid.values[r][c] == 8:  # Sky blue
                    new_grid.values[r][c] = 8
                elif template[r][c] == 2:  # Fixed zone
                    new_grid.values[r][c] = 7  # Orange
                elif template[r][c] == 0:
                    new_grid.values[r][c] = 1  # Blue
                else:
                    new_grid.values[r][c] = 7  # Orange
    return new_grid

def balance_colors(grid: ColoredGrid) -> ColoredGrid:
    color_count = {1: 0, 7: 0, 8: 0}
    for row in grid.values:
        for cell in row:
            if cell in color_count:
                color_count[cell] += 1
    
    if abs(color_count[1] - color_count[7]) > 5:  # Arbitrary threshold
        majority_color = 1 if color_count[1] > color_count[7] else 7
        minority_color = 7 if majority_color == 1 else 1
        diff = abs(color_count[1] - color_count[7]) // 2
        
        for r in range(len(grid.values)):
            for c in range(len(grid.values[0])):
                if grid.values[r][c] == majority_color and diff > 0:
                    grid.values[r][c] = minority_color
                    diff -= 1
    return grid

def ensure_contrast(grid: ColoredGrid) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0:
                neighbors = get_neighbors(grid, r, c)
                if all(neighbor == grid.values[r][c] for neighbor in neighbors if neighbor != 0):
                    grid.values[r][c] = 1 if grid.values[r][c] == 7 else 7
    return grid

def get_neighbors(grid: ColoredGrid, r: int, c: int) -> List[int]:
    rows, cols = grid.get_dimensions()
    neighbors = []
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append(grid.values[nr][nc])
    return neighbors
