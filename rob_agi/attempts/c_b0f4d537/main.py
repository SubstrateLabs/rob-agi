from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b0f4d537(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies vertical lines on both sides of a gray (5) dividing line in the input grid.
    2. Creates a new 7-column wide grid with the same height as the input.
    3. Maps vertical lines from input to output, scaling their positions.
    4. Processes each row:
       - Full horizontal lines are preserved across all columns.
       - Other cells are filled based on the nearest non-zero, non-gray color in the input.
    5. Ensures vertical lines are continuous from top to bottom.
    6. Handles special cases where no vertical lines are found.

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

    def get_vertical_lines(grid: ColoredGrid, dividing_line: int) -> List[Tuple[int, int]]:
        lines = []
        rows, cols = grid.get_dimensions()
        for c in range(cols):
            if c != dividing_line:
                color = next((grid.get_cell(r, c) for r in range(rows) if grid.get_cell(r, c) not in [0, 5]), None)
                if color:
                    relative_pos = c / dividing_line if c < dividing_line else (c - dividing_line) / (cols - dividing_line - 1)
                    lines.append((color, relative_pos))
        return lines

    def map_vertical_lines(lines: List[Tuple[int, int]], output_width: int) -> List[Tuple[int, int]]:
        return [(color, int(pos * (output_width - 1))) for color, pos in lines]

    def process_row(input_row: List[int], vertical_lines: List[Tuple[int, int]], output_width: int) -> List[int]:
        output_row = [0] * output_width
        non_zero_colors = [c for c in input_row if c not in [0, 5]]
        
        if len(set(non_zero_colors)) == 1 and non_zero_colors:
            return [non_zero_colors[0]] * output_width
        
        for i in range(output_width):
            nearest_color = 0
            nearest_distance = float('inf')
            for color, pos in vertical_lines:
                if abs(i - pos) < nearest_distance:
                    nearest_color = color
                    nearest_distance = abs(i - pos)
            
            input_pos = int(i * len(input_row) / output_width)
            if input_row[input_pos] not in [0, 5]:
                output_row[i] = input_row[input_pos]
            else:
                output_row[i] = nearest_color
        
        return output_row

    rows, cols = input_grid.get_dimensions()
    dividing_line = find_dividing_line(input_grid)
    vertical_lines = get_vertical_lines(input_grid, dividing_line)
    
    if not vertical_lines:
        most_common_color = max(set(input_grid.values[r][c] for r in range(rows) for c in range(cols) if input_grid.values[r][c] not in [0, 5]), key=lambda x: sum(row.count(x) for row in input_grid.values))
        vertical_lines = [(most_common_color, 0.5)]
    
    output_vertical_lines = map_vertical_lines(vertical_lines, 7)
    
    output_values = []
    for r in range(rows):
        input_row = [input_grid.get_cell(r, c) for c in range(cols) if c != dividing_line]
        output_row = process_row(input_row, output_vertical_lines, 7)
        output_values.append(output_row)
    
    # Ensure vertical lines are continuous
    for color, pos in output_vertical_lines:
        for r in range(rows):
            if output_values[r][pos] == 0:
                output_values[r][pos] = color

    return ColoredGrid(values=output_values)
