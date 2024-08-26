from rob_agi.colored_grid import ColoredGrid

def create_backwards_c_grid() -> ColoredGrid:
    return ColoredGrid(values=[[0,0,8],[8,8,0],[0,8,0]])

def create_inverted_l_grid() -> ColoredGrid:
    return ColoredGrid(values=[[8,8,0],[8,0,0],[8,0,0]])

def create_horizontal_stripe_grid() -> ColoredGrid:
    return ColoredGrid(values=[[0,0,0],[8,8,8],[0,0,0]])

def solve_9110e3c5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 7x7 input grid into a 3x3 output grid based on the distribution of black cells.
    
    The function analyzes the distribution of black cells (0) in the input grid,
    particularly in the center-right area and upper-right quadrant. Based on this
    distribution, it returns one of three predefined 3x3 patterns:
    1. Backwards "C" pattern
    2. Inverted "L" pattern
    3. Horizontal stripe pattern

    Args:
    input_grid (ColoredGrid): A 7x7 ColoredGrid object representing the input.

    Returns:
    ColoredGrid: A 3x3 ColoredGrid object representing the output pattern.
    """
    rows, cols = input_grid.get_dimensions()
    center_right_black = 0
    upper_right_black = 0
    total_black = 0

    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 0:  # Black cell
                total_black += 1
                if c >= cols // 2:
                    center_right_black += 1
                    if r < rows // 2:
                        upper_right_black += 1

    if total_black == 0:
        return create_horizontal_stripe_grid()

    center_right_ratio = center_right_black / total_black
    upper_right_ratio = upper_right_black / total_black

    CENTER_RIGHT_THRESHOLD = 0.5
    UPPER_RIGHT_THRESHOLD = 0.3

    if center_right_ratio >= CENTER_RIGHT_THRESHOLD:
        return create_backwards_c_grid()
    elif upper_right_ratio >= UPPER_RIGHT_THRESHOLD:
        return create_inverted_l_grid()
    else:
        return create_horizontal_stripe_grid()
