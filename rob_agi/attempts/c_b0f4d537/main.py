from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b0f4d537(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies the gray (5) dividing line in the input grid.
    2. Extracts vertical color patterns from both sides of the dividing line.
    3. Creates a new 7-column wide grid with the same height as the input.
    4. Maps vertical patterns to the output grid, scaling their positions.
    5. Processes each row:
       - Full horizontal lines are preserved across all columns.
       - Other cells are filled based on the nearest vertical pattern.
    6. Ensures vertical patterns are continuous from top to bottom.
    7. Handles special cases where no vertical patterns are found.

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
                    width = 1
                    while c + width < cols and c + width != dividing_line and grid.get_cell(0, c + width) == color:
                        width += 1
                    relative_pos = c / dividing_line if c < dividing_line else (c - dividing_line) / (cols - dividing_line - 1)
                    patterns.append((color, relative_pos, width))
                    c += width - 1
        return patterns

    def map_vertical_patterns(patterns: List[Tuple[int, float, int]], output_width: int) -> List[Tuple[int, int, int]]:
        return [(color, int(pos * (output_width - 1)), max(1, int(width * output_width / (output_width - 1)))) for color, pos, width in patterns]

    def process_row(input_row: List[int], vertical_patterns: List[Tuple[int, int, int]], output_width: int) -> List[int]:
        output_row = [0] * output_width
        non_zero_colors = [c for c in input_row if c not in [0, 5]]
        
        if len(set(non_zero_colors)) == 1 and non_zero_colors:
            return [non_zero_colors[0]] * output_width
        
        for i in range(output_width):
            nearest_color = 0
            nearest_distance = float('inf')
            for color, pos, width in vertical_patterns:
                if pos <= i < pos + width:
                    nearest_color = color
                    break
                if abs(i - pos) < nearest_distance:
                    nearest_color = color
                    nearest_distance = abs(i - pos)
            output_row[i] = nearest_color
        
        return output_row

    rows, cols = input_grid.get_dimensions()
    dividing_line = find_dividing_line(input_grid)
    vertical_patterns = get_vertical_patterns(input_grid, dividing_line)
    
    if not vertical_patterns:
        most_common_color = max(set(input_grid.values[r][c] for r in range(rows) for c in range(cols) if input_grid.values[r][c] not in [0, 5]), key=lambda x: sum(row.count(x) for row in input_grid.values))
        vertical_patterns = [(most_common_color, 0.5, 1)]
    
    output_vertical_patterns = map_vertical_patterns(vertical_patterns, 7)
    
    output_values = []
    for r in range(rows):
        input_row = [input_grid.get_cell(r, c) for c in range(cols) if c != dividing_line]
        output_row = process_row(input_row, output_vertical_patterns, 7)
        output_values.append(output_row)
    
    # Ensure vertical patterns are continuous
    for color, pos, width in output_vertical_patterns:
        for r in range(rows):
            for c in range(pos, min(pos + width, 7)):
                if output_values[r][c] == 0:
                    output_values[r][c] = color

    return ColoredGrid(values=output_values)
