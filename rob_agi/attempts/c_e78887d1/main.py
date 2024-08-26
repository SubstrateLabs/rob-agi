from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e78887d1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 3-row representation that captures the essence of the input patterns.
    
    The function performs the following steps:
    1. Identifies distinct color groups and their patterns in the input grid.
    2. Creates an idealized 3-row representation for each color group.
    3. Combines the representations while maintaining relative proportions and order.
    4. Refines the output to ensure balance and utilization of all 3 rows.
    
    This approach focuses on distilling and idealizing the essence of the input patterns
    rather than strict replication, allowing for creative interpretation while maintaining
    consistency across different inputs.
    """
    rows, cols = input_grid.get_dimensions()
    color_groups = identify_color_groups(input_grid)
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(3)])
    
    for color, positions in color_groups.items():
        pattern = identify_pattern(positions, rows, cols)
        representation = create_representation(color, pattern, cols)
        merge_representation(output_grid, representation)
    
    refine_output(output_grid)
    
    return output_grid

def identify_color_groups(grid: ColoredGrid) -> Dict[int, List[Tuple[int, int]]]:
    color_groups = {}
    for r in range(grid.get_dimensions()[0]):
        for c in range(grid.get_dimensions()[1]):
            color = grid.values[r][c]
            if color != 0:
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append((r, c))
    return color_groups

def identify_pattern(positions: List[Tuple[int, int]], rows: int, cols: int) -> str:
    if len(set(r for r, _ in positions)) == 1:
        return "horizontal"
    if len(set(c for _, c in positions)) == 1:
        return "vertical"
    if len(positions) >= rows * cols / 4:
        return "block"
    return "scattered"

def create_representation(color: int, pattern: str, cols: int) -> List[List[int]]:
    if pattern == "vertical":
        return [[color, 0, color] for _ in range(3)]
    if pattern == "horizontal":
        return [[0, 0, 0], [color] * 3, [0, 0, 0]]
    if pattern == "block":
        return [[color, color, 0], [color, color, color], [0, color, color]]
    return [[color, 0, 0], [0, color, 0], [0, 0, color]]  # scattered

def merge_representation(output_grid: ColoredGrid, representation: List[List[int]]):
    for r in range(3):
        for c in range(len(representation[0])):
            if representation[r][c] != 0:
                output_grid.values[r][output_grid.values[r].index(0)] = representation[r][c]

def refine_output(grid: ColoredGrid):
    for col in range(grid.get_dimensions()[1]):
        colors = [grid.values[r][col] for r in range(3)]
        if len(set(colors)) == 1 and colors[0] != 0:
            grid.values[1][col] = 0
        elif colors.count(0) == 2:
            non_zero = next(color for color in colors if color != 0)
            for r in range(3):
                if grid.values[r][col] == 0:
                    grid.values[r][col] = non_zero
                    break

def complete_patterns(grid: ColoredGrid):
    """Completes patterns in the grid by filling in missing parts of shapes and ensuring consistency."""
    for col in range(grid.get_dimensions()[1]):
        if is_partial_vertical_line(grid, col):
            complete_vertical_line(grid, col)
        if is_partial_horizontal_line(grid, col):
            complete_horizontal_line(grid, col)
    
    # Ensure vertical alignment
    for col in range(grid.get_dimensions()[1]):
        align_vertically(grid, col)

def align_vertically(grid: ColoredGrid, col: int):
    """Aligns colors vertically in a column, moving non-zero values to the top."""
    colors = [grid.values[row][col] for row in range(3) if grid.values[row][col] != 0]
    for row in range(len(colors)):
        grid.values[row][col] = colors[row]
    for row in range(len(colors), 3):
        grid.values[row][col] = 0

def is_partial_vertical_line(grid: ColoredGrid, col: int) -> bool:
    """Checks if there's a partial vertical line in the given column."""
    colors = [grid.values[row][col] for row in range(3)]
    return colors.count(0) == 1 and len(set(colors) - {0}) == 1

def complete_vertical_line(grid: ColoredGrid, col: int):
    """Completes a partial vertical line in the given column."""
    color = max(set(grid.values[row][col] for row in range(3)) - {0})
    for row in range(3):
        if grid.values[row][col] == 0:
            grid.values[row][col] = color

def is_partial_horizontal_line(grid: ColoredGrid, col: int) -> bool:
    """Checks if there's a partial horizontal line starting from the given column."""
    return any(sum(1 for c in range(col, min(col+3, grid.get_dimensions()[1])) if grid.values[row][c] == color) == 2
               for row in range(3)
               for color in set(grid.values[row][c] for c in range(col, min(col+3, grid.get_dimensions()[1]))) - {0})

def complete_horizontal_line(grid: ColoredGrid, col: int):
    """Completes a partial horizontal line starting from the given column."""
    for row in range(3):
        colors = [grid.values[row][c] for c in range(col, min(col+3, grid.get_dimensions()[1]))]
        if len(set(colors) - {0}) == 1 and colors.count(0) == 1:
            color = max(set(colors) - {0})
            for c in range(col, min(col+3, grid.get_dimensions()[1])):
                if grid.values[row][c] == 0:
                    grid.values[row][c] = color
