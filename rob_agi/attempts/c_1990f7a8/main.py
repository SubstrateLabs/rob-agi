from rob_agi.colored_grid import ColoredGrid

def solve_1990f7a8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid into a 7x7 output grid by analyzing patterns in each quadrant.
    
    The function divides the input into four quadrants, extracts the key pattern from each,
    and assembles these into a 7x7 grid. It preserves the essence of red (2) patterns
    while maintaining a black (0) separator row in the middle.
    
    Steps:
    1. Divide input into quadrants
    2. Extract and simplify the pattern from each quadrant
    3. Create a 3x3 representation for each quadrant
    4. Assemble the 3x3 representations into a 7x7 output grid
    5. Handle the middle column to connect patterns
    6. Ensure the middle row (row 3) remains black
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: A 7x7 grid representing the transformed input
    """
    rows, cols = input_grid.get_dimensions()
    mid_row, mid_col = rows // 2, cols // 2

    def extract_pattern(top, left, bottom, right):
        subgrid = input_grid.extract_subgrid(top, left, bottom - top + 1, right - left + 1)
        pattern = [[0 for _ in range(3)] for _ in range(3)]
        
        # Find the bounding box of the red cells
        red_cells = [(r, c) for r in range(bottom - top + 1) for c in range(right - left + 1) if subgrid.get_cell(r, c) == 2]
        if not red_cells:
            return pattern
        
        min_r, min_c = min(red_cells)
        max_r, max_c = max(red_cells)
        
        # Map the pattern to 3x3
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if subgrid.get_cell(r, c) == 2:
                    pattern_r = min(2, (r - min_r) * 3 // (max_r - min_r + 1))
                    pattern_c = min(2, (c - min_c) * 3 // (max_c - min_c + 1))
                    pattern[pattern_r][pattern_c] = 2
        
        return pattern

    quadrants = [
        extract_pattern(0, 0, mid_row - 1, mid_col - 1),
        extract_pattern(0, mid_col, mid_row - 1, cols - 1),
        extract_pattern(mid_row, 0, rows - 1, mid_col - 1),
        extract_pattern(mid_row, mid_col, rows - 1, cols - 1)
    ]

    output_values = [[0 for _ in range(7)] for _ in range(7)]
    
    for i, quad in enumerate(quadrants):
        start_row = 0 if i < 2 else 4
        start_col = 0 if i % 2 == 0 else 4
        for r in range(3):
            for c in range(3):
                output_values[start_row + r][start_col + c] = quad[r][c]

    # Handle middle column
    for r in range(3):
        if output_values[r][2] == 2 or output_values[r][4] == 2:
            output_values[r][3] = 2
    for r in range(4, 7):
        if output_values[r][2] == 2 or output_values[r][4] == 2:
            output_values[r][3] = 2

    # Ensure middle row is black
    output_values[3] = [0, 0, 0, 0, 0, 0, 0]

    return ColoredGrid(values=output_values)
