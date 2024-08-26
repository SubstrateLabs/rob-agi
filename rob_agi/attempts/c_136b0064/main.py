from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_136b0064(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing shapes and reorganizing them into a 7-column grid.
    
    1. Analyzes the input grid to find the yellow line and gray cell.
    2. Extracts and compresses shapes, maintaining their vertical structure.
    3. Places shapes in a 7-column output grid, preserving relative positions.
    4. Handles the gray cell placement based on its original position.
    5. Balances the composition and optimizes the layout.
    6. Adjusts the final grid to meet size constraints.
    
    Returns a new ColoredGrid with the transformed layout.
    """
    rows, cols = input_grid.get_dimensions()
    yellow_col = next(c for c in range(cols) if input_grid.values[0][c] == 4)
    gray_pos = next(((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 5), None)
    
    def extract_shape(start_col):
        shape = []
        for r in range(rows):
            col_slice = [input_grid.values[r][c] for c in range(start_col, yellow_col)]
            if any(color != 0 for color in col_slice):
                shape.append(col_slice)
        return [col for col in zip(*shape) if any(color != 0 for color in col)]
    
    shapes = []
    col = 0
    while col < yellow_col:
        if input_grid.values[0][col] != 0:
            shape = extract_shape(col)
            shapes.append(shape)
            col += len(shape)
        else:
            col += 1
    
    output = [[0 for _ in range(7)] for _ in range(15)]
    
    # Place gray cell
    if gray_pos:
        gray_col = min(6, max(0, int(gray_pos[1] * 7 / cols)))
        output[0][gray_col] = 5
    
    # Place shapes
    output_col = 6
    for shape in reversed(shapes):
        shape_width = len(shape)
        for c, column in enumerate(shape):
            for r, color in enumerate(column):
                if color != 0:
                    output[r][output_col - shape_width + c + 1] = color
        output_col = max(0, output_col - shape_width)
    
    # Optimize layout
    output = [row for row in output if any(cell != 0 for cell in row)]
    while len(output) < 7:
        output.append([0] * 7)
    output = output[:7]
    
    # Shift non-black cells to the right in each row
    for r in range(len(output)):
        non_black = [c for c in output[r] if c != 0]
        output[r] = [0] * (7 - len(non_black)) + non_black
    
    return ColoredGrid(values=output)
