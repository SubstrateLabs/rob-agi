from rob_agi.colored_grid import ColoredGrid

def create_backwards_c_grid() -> ColoredGrid:
    return ColoredGrid(values=[[0,8,8],[0,8,0],[0,8,0]])

def create_inverted_l_grid() -> ColoredGrid:
    return ColoredGrid(values=[[8,8,0],[8,0,0],[8,0,0]])

def create_horizontal_stripe_grid() -> ColoredGrid:
    return ColoredGrid(values=[[0,0,0],[8,8,8],[0,0,0]])

def solve_9110e3c5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 7x7 input grid into a 3x3 output grid based on the distribution of non-black cells.
    
    The function analyzes the pattern of non-black cells in the input grid,
    focusing on the overall distribution and presence of vertical lines.
    Based on this analysis, it returns one of three predefined 3x3 patterns:
    1. Backwards "C" pattern (strong right vertical line or right-heavy distribution)
    2. Inverted "L" pattern (strong left vertical line or left-heavy distribution)
    3. Horizontal stripe pattern (balanced distribution or no strong vertical lines)

    Args:
    input_grid (ColoredGrid): A 7x7 ColoredGrid object representing the input.

    Returns:
    ColoredGrid: A 3x3 ColoredGrid object representing the output pattern.
    """
    rows, cols = input_grid.get_dimensions()
    left_half_count = 0
    right_half_count = 0
    left_vertical_line = 0
    right_vertical_line = 0

    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) != 0:  # Non-black cell
                if c < cols // 2:
                    left_half_count += 1
                else:
                    right_half_count += 1
                if c == 1:
                    left_vertical_line += 1
                if c == cols - 2:
                    right_vertical_line += 1

    # Check for strong vertical lines
    if right_vertical_line >= 4:
        return create_backwards_c_grid()
    elif left_vertical_line >= 4:
        return create_inverted_l_grid()

    # Check overall distribution
    if right_half_count > left_half_count * 1.5:
        return create_backwards_c_grid()
    elif left_half_count > right_half_count * 1.5:
        return create_inverted_l_grid()

    # Default to horizontal stripe if no clear pattern is found
    return create_horizontal_stripe_grid()
