from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_782b5218(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on color weights and pattern detection.
    
    1. Analyzes the input grid to calculate color weights.
    2. Determines if the pattern should be horizontal banding or diagonal.
    3. Creates a new grid with colors arranged based on their weights.
    4. For horizontal banding: fills from top to bottom with sorted colors.
    5. For diagonal pattern: fills diagonally from top-left to bottom-right.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    rows, cols = input_grid.get_dimensions()
    color_weights = calculate_color_weights(input_grid)
    sorted_colors = sorted(color_weights.keys(), key=lambda c: color_weights[c])
    
    if has_uniform_middle_row(input_grid):
        return create_horizontal_banding(input_grid, sorted_colors)
    else:
        return create_diagonal_pattern(input_grid, sorted_colors)

def calculate_color_weights(grid: ColoredGrid) -> Dict[int, float]:
    weights = {}
    for r, row in enumerate(grid.values):
        for color in row:
            if color not in weights:
                weights[color] = []
            weights[color].append(r)
    return {color: sum(rows) / len(rows) for color, rows in weights.items()}

def has_uniform_middle_row(grid: ColoredGrid) -> bool:
    middle_row = len(grid.values) // 2
    return len(set(grid.values[middle_row])) == 1

def create_horizontal_banding(grid: ColoredGrid, sorted_colors: List[int]) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    new_values = [[0 for _ in range(cols)] for _ in range(rows)]
    
    middle_row = rows // 2
    top_color = sorted_colors[0]
    bottom_color = sorted_colors[-1]
    
    for r in range(rows):
        if r < middle_row:
            new_values[r] = [top_color] * cols
        elif r == middle_row and has_uniform_middle_row(grid):
            new_values[r] = grid.values[r]
        else:
            new_values[r] = [bottom_color] * cols
    
    return ColoredGrid(values=new_values)

def create_diagonal_pattern(grid: ColoredGrid, sorted_colors: List[int]) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    new_values = [[0 for _ in range(cols)] for _ in range(rows)]
    num_colors = len(sorted_colors)
    diagonal_width = (rows + cols) // num_colors
    
    for r in range(rows):
        for c in range(cols):
            diagonal_index = (r + c) // diagonal_width
            color_index = min(diagonal_index, num_colors - 1)
            new_values[r][c] = sorted_colors[color_index]
    
    return ColoredGrid(values=new_values)
