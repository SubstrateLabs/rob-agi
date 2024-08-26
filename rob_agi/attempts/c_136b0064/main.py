from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_136b0064(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing shapes and reorganizing them into a 7-column grid.
    
    1. Analyzes the input grid to find the yellow line and gray cell.
    2. Extracts and compresses shapes, maintaining their vertical structure.
    3. Places shapes in a 7-column output grid, preserving relative positions.
    4. Handles the gray cell placement at the top of the grid.
    5. Right-aligns non-empty cells in each row.
    6. Adjusts the final grid to meet size constraints (7x7).
    
    Returns a new ColoredGrid with the transformed layout.
    """
    rows, cols = input_grid.get_dimensions()
    yellow_col = next(c for c in range(cols) if input_grid.values[0][c] == 4)
    gray_pos = next(((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 5), None)
    
    def extract_shape(start_col):
        shape = []
        for r in range(rows):
            col_slice = [input_grid.values[r][c] for c in range(start_col, yellow_col)]
            shape.append(col_slice)
        return [col for col in zip(*shape) if any(color != 0 for color in col)]
    
    shapes = []
    col = 0
    while col < yellow_col:
        if any(input_grid.values[r][col] != 0 for r in range(rows)):
            shape = extract_shape(col)
            shapes.append(shape)
            col += len(shape)
        else:
            col += 1
    
    output = [[0 for _ in range(7)] for _ in range(15)]
    
    # Place shapes
    output_row = 14
    for shape in shapes:
        shape_height = len(shape[0])
        shape_width = len(shape)
        for c, column in enumerate(shape):
            for r, color in enumerate(column):
                if color != 0:
                    output[output_row - shape_height + r + 1][7 - shape_width + c] = color
        output_row = max(0, output_row - shape_height)
    
    # Place gray cell
    if gray_pos:
        gray_col = min(6, max(0, int(gray_pos[1] * 7 / cols)))
        output[0][gray_col] = 5
    
    # Remove empty rows and ensure 7x7 grid
    output = [row for row in output if any(cell != 0 for cell in row)]
    while len(output) < 7:
        output.insert(0, [0] * 7)
    output = output[:7]
    
    # Right-align non-black cells in each row
    for r in range(len(output)):
        non_black = [c for c in output[r] if c != 0]
        output[r] = [0] * (7 - len(non_black)) + non_black
    
    return ColoredGrid(values=output)
