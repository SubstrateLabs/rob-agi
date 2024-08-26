from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_136b0064(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing shapes and reorganizing them into a 7-column grid.
    
    1. Analyzes the input grid to find the yellow line and gray cell.
    2. Creates vertical slices of the grid, compressing them horizontally.
    3. Places compressed slices in a new 7-column grid, maintaining vertical order.
    4. Handles special cases like gray cells and optimizes the layout.
    5. Fills remaining space with black (empty) cells.
    
    Returns a new ColoredGrid with the transformed layout.
    """
    rows, cols = input_grid.get_dimensions()
    yellow_col = next(c for c in range(cols) if input_grid.values[0][c] == 4)
    gray_pos = next(((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 5), None)
    
    def create_slices():
        slices = []
        for c in range(cols):
            if c != yellow_col:
                slice_colors = [input_grid.values[r][c] for r in range(rows)]
                slices.append((slice_colors, c < yellow_col, c))
        return slices
    
    def compress_slice(slice_colors):
        compressed = []
        for color in slice_colors:
            if color != 0 or not compressed or compressed[-1] != 0:
                compressed.append(color)
        return compressed if len(compressed) <= 3 else compressed[:3]
    
    slices = create_slices()
    compressed_slices = [(compress_slice(colors), is_left, orig_pos) for colors, is_left, orig_pos in slices]
    
    output = [[0 for _ in range(7)] for _ in range(rows)]
    left_col, right_col = 0, 6
    
    # Place right side slices
    for colors, is_left, _ in sorted(compressed_slices, key=lambda x: (not x[1], x[2])):
        if not is_left:
            for r, color in enumerate(colors):
                if color != 0:
                    output[r][right_col] = color
            right_col -= 1
    
    # Place left side slices
    for colors, is_left, _ in sorted(compressed_slices, key=lambda x: (x[1], -x[2])):
        if is_left:
            for r, color in enumerate(colors):
                if color != 0:
                    output[r][left_col] = color
            left_col += 1
    
    # Handle gray cell
    if gray_pos:
        gray_distance = cols - gray_pos[1] - 1
        output[0][6 - gray_distance] = 5
    
    # Optimize layout
    output = [row for row in output if any(cell != 0 for cell in row)]
    while len(output) < 7:
        output.append([0] * 7)
    
    # Shift non-black columns to the right
    for r in range(len(output)):
        output[r] = [0] * (7 - sum(1 for c in output[r] if c != 0)) + [c for c in output[r] if c != 0]
    
    return ColoredGrid(values=output)
