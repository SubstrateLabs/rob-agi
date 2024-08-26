from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b0f4d537(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies the gray (5) dividing line in the input grid.
    2. Extracts vertical color patterns from both sides of the dividing line.
    3. Identifies full horizontal lines of a single color.
    4. Creates a new 7-column wide grid with the same height as the input.
    5. Maps vertical patterns to the output grid, preserving their relative positions.
    6. Applies horizontal lines to the output grid.
    7. Fills remaining cells based on the nearest vertical pattern.
    8. Ensures vertical patterns are continuous from top to bottom.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed output grid.
    """
    def find_dividing_line(grid: ColoredGrid) -> int:
        rows, cols = grid.get_dimensions()
        for c in range(cols):
            if all(grid.get_cell(r, c) == 5 for r in range(rows)):
                return c
        return cols // 2  # Default to middle if no dividing line found

    def get_vertical_patterns(grid: ColoredGrid, dividing_line: int) -> List[Tuple[int, float, int]]:
        patterns = []
        rows, cols = grid.get_dimensions()
        for c in range(cols):
            if c != dividing_line:
                colors = [grid.get_cell(r, c) for r in range(rows) if grid.get_cell(r, c) not in [0, 5]]
                if colors:
                    color = max(set(colors), key=colors.count)
                    relative_pos = c / dividing_line if c < dividing_line else (c - dividing_line - 1) / (cols - dividing_line - 1)
                    patterns.append((color, relative_pos))
        return patterns

    def get_horizontal_lines(grid: ColoredGrid, dividing_line: int) -> List[Tuple[int, int]]:
        rows, cols = grid.get_dimensions()
        horizontal_lines = []
        for r in range(rows):
            colors = set(grid.get_cell(r, c) for c in range(cols) if c != dividing_line and grid.get_cell(r, c) not in [0, 5])
            if len(colors) == 1:
                horizontal_lines.append((r, list(colors)[0]))
        return horizontal_lines

    def map_vertical_patterns(patterns: List[Tuple[int, float]], output_width: int) -> List[Tuple[int, int]]:
        return [(color, int(pos * (output_width - 1))) for color, pos in patterns]

    rows, cols = input_grid.get_dimensions()
    dividing_line = find_dividing_line(input_grid)
    vertical_patterns = get_vertical_patterns(input_grid, dividing_line)
    horizontal_lines = get_horizontal_lines(input_grid, dividing_line)
    
    if not vertical_patterns:
        most_common_color = max(set(input_grid.values[r][c] for r in range(rows) for c in range(cols) if input_grid.values[r][c] not in [0, 5]), key=lambda x: sum(row.count(x) for row in input_grid.values))
        vertical_patterns = [(most_common_color, 0.5)]
    
    output_vertical_patterns = map_vertical_patterns(vertical_patterns, 7)
    
    output_values = [[0 for _ in range(7)] for _ in range(rows)]
    
    # Apply horizontal lines
    for r, color in horizontal_lines:
        output_values[r] = [color] * 7
    
    # Apply vertical patterns
    for color, pos in output_vertical_patterns:
        for r in range(rows):
            if output_values[r][pos] == 0:
                output_values[r][pos] = color
    
    # Fill remaining cells
    for r in range(rows):
        if output_values[r][0] == 0:  # Not a horizontal line
            for c in range(7):
                if output_values[r][c] == 0:
                    nearest_pattern = min(output_vertical_patterns, key=lambda x: abs(x[1] - c))
                    output_values[r][c] = nearest_pattern[0]
    
    # Ensure vertical patterns are continuous
    for color, pos in output_vertical_patterns:
        for r in range(1, rows):
            if output_values[r-1][pos] == color and output_values[r][pos] != color:
                output_values[r][pos] = color

    return ColoredGrid(values=output_values)
