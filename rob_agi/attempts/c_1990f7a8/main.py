from rob_agi.colored_grid import ColoredGrid

def solve_1990f7a8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid into a 7x7 output grid by analyzing patterns in each quadrant.
    
    The function divides the input into four quadrants, extracts the key pattern from each,
    and assembles these into a 7x7 grid. It preserves the exact patterns when possible,
    simplifies larger patterns, and maintains connectivity between quadrants.
    
    Steps:
    1. Divide input into quadrants
    2. Extract and simplify the pattern from each quadrant, considering the center of mass
    3. Create a 3x3 representation for each quadrant, preserving exact patterns when possible
    4. Assemble the 3x3 representations into a 7x7 output grid
    5. Handle the middle column to connect patterns across quadrants
    6. Ensure the middle row (row 3) remains black as a separator
    7. Make final adjustments to ensure pattern representation and connectivity
    
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
        
        red_cells = [(r, c) for r in range(bottom - top + 1) for c in range(right - left + 1) if subgrid.get_cell(r, c) == 2]
        if not red_cells:
            return pattern
        
        # Calculate center of mass
        center_r = sum(r for r, _ in red_cells) / len(red_cells)
        center_c = sum(c for _, c in red_cells) / len(red_cells)
        
        height, width = bottom - top + 1, right - left + 1
        
        if height <= 3 and width <= 3:
            # If pattern fits within 3x3, preserve it exactly
            for r, c in red_cells:
                pattern[r][c] = 2
        else:
            # Simplify larger patterns
            for r in range(3):
                for c in range(3):
                    r_start, r_end = top + r * height // 3, top + (r + 1) * height // 3
                    c_start, c_end = left + c * width // 3, left + (c + 1) * width // 3
                    if any(r_start <= rr < r_end and c_start <= cc < c_end for rr, cc in red_cells):
                        pattern[r][c] = 2
        
        # Ensure at least one red cell if original had red cells
        if sum(sum(row) for row in pattern) == 0 and red_cells:
            nearest_r = min(range(3), key=lambda r: abs(r - center_r * 3 / height))
            nearest_c = min(range(3), key=lambda c: abs(c - center_c * 3 / width))
            pattern[nearest_r][nearest_c] = 2
        
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

    # Final adjustments
    for r in range(7):
        if r != 3 and sum(output_values[r]) == 0:
            if r < 3 and sum(sum(row) for row in quadrants[0 if r < 3 else 2]) > 0:
                output_values[r][1] = 2
            elif r > 3 and sum(sum(row) for row in quadrants[1 if r < 3 else 3]) > 0:
                output_values[r][5] = 2

    return ColoredGrid(values=output_values)
