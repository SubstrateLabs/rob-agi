from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_136b0064(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing shapes and reorganizing them into a 7-column grid.
    
    1. Analyzes the input grid to find the yellow line and gray cell.
    2. Compresses the grid horizontally, maintaining vertical integrity of shapes.
    3. Maps compressed columns to a 7-column output grid.
    4. Handles special cases like the gray cell, ensuring correct positioning.
    5. Optimizes the layout and fills remaining space with black (empty) cells.
    
    Returns a new ColoredGrid with the transformed layout.
    """
    rows, cols = input_grid.get_dimensions()
    yellow_col = next(c for c in range(cols) if input_grid.values[0][c] == 4)
    gray_pos = next(((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 5), None)
    
    def compress_column(col):
        return [color for color in (input_grid.values[r][col] for r in range(rows)) if color != 0]
    
    compressed_columns = [compress_column(c) for c in range(cols) if c != yellow_col]
    
    output_width = 7
    compression_ratio = (cols - 1) / output_width
    
    output = [[0 for _ in range(output_width)] for _ in range(min(15, rows))]
    
    for i, column in enumerate(compressed_columns):
        output_col = min(output_width - 1, int((i + 0.5) * compression_ratio))
        for r, color in enumerate(column):
            if r < len(output):
                output[r][output_col] = color
    
    if gray_pos:
        gray_distance = cols - gray_pos[1] - 1
        output_gray_col = min(output_width - 1, output_width - 1 - int(gray_distance / compression_ratio))
        output[0][output_gray_col] = 5
    
    # Optimize layout
    output = [row for row in output if any(cell != 0 for cell in row)]
    while len(output) < 7:
        output.append([0] * output_width)
    
    # Shift non-black cells to the right in each row
    for r in range(len(output)):
        non_black = [c for c in output[r] if c != 0]
        output[r] = [0] * (output_width - len(non_black)) + non_black
    
    return ColoredGrid(values=output)
