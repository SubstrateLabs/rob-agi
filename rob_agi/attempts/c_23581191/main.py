from rob_agi.colored_grid import ColoredGrid

def solve_23581191(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending the colors 8 and 7 in specific patterns:
    - For 8: Fills its entire column and row to the right
    - For 7: Fills its entire column and row to the left
    - Places 2 at the intersections of these extensions
    - Preserves the original positions of 8 and 7

    Args:
    input_grid (ColoredGrid): The input grid to transform

    Returns:
    ColoredGrid: The transformed grid
    """
    rows, cols = input_grid.get_dimensions()
    output = input_grid.deep_copy()
    
    def find_number(num):
        for r in range(rows):
            for c in range(cols):
                if input_grid.get_cell(r, c) == num:
                    return r, c
        return None, None

    r8, c8 = find_number(8)
    r7, c7 = find_number(7)

    if r8 is not None and c8 is not None:
        for r in range(rows):
            output.set_cell(r, c8, 8)
        for c in range(c8, cols):
            output.set_cell(r8, c, 8)

    if r7 is not None and c7 is not None:
        for r in range(rows):
            output.set_cell(r, c7, 7)
        for c in range(c7 + 1):
            output.set_cell(r7, c, 7)

    if r8 is not None and c8 is not None and r7 is not None and c7 is not None:
        output.set_cell(r7, c8, 2)
        output.set_cell(r8, c7, 2)

    # Preserve original positions
    if r8 is not None and c8 is not None:
        output.set_cell(r8, c8, 8)
    if r7 is not None and c7 is not None:
        output.set_cell(r7, c7, 7)

    return output
