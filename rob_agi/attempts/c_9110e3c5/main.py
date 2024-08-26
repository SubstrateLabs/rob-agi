from rob_agi.colored_grid import ColoredGrid

def create_backwards_c_grid() -> ColoredGrid:
    return ColoredGrid(values=[[0,0,8],[8,8,0],[0,8,0]])

def create_inverted_l_grid() -> ColoredGrid:
    return ColoredGrid(values=[[8,8,0],[8,0,0],[8,0,0]])

def create_horizontal_stripe_grid() -> ColoredGrid:
    return ColoredGrid(values=[[0,0,0],[8,8,8],[0,0,0]])

def solve_9110e3c5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 7x7 input grid into a 3x3 output grid based on the distribution of non-black cells.
    
    The function analyzes the distribution of non-black cells in the input grid,
    particularly in the center and right half areas. Based on this distribution,
    it returns one of three predefined 3x3 patterns:
    1. Backwards "C" pattern
    2. Inverted "L" pattern
    3. Horizontal stripe pattern

    Args:
    input_grid (ColoredGrid): A 7x7 ColoredGrid object representing the input.

    Returns:
    ColoredGrid: A 3x3 ColoredGrid object representing the output pattern.
    """
    rows, cols = input_grid.get_dimensions()
    total_non_black = 0
    center_non_black = 0
    right_half_non_black = 0

    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) != 0:  # Non-black cell
                total_non_black += 1
                if 2 <= r <= 4 and 2 <= c <= 4:  # Center 3x3 area
                    center_non_black += 1
                if c >= cols // 2:  # Right half
                    right_half_non_black += 1

    if total_non_black == 0:
        return create_horizontal_stripe_grid()

    center_density = center_non_black / total_non_black
    right_half_density = right_half_non_black / total_non_black

    CENTER_THRESHOLD = 0.2
    RIGHT_HALF_THRESHOLD = 0.6

    if center_density < CENTER_THRESHOLD:
        if right_half_density > RIGHT_HALF_THRESHOLD:
            return create_backwards_c_grid()
        else:
            return create_horizontal_stripe_grid()
    else:
        return create_inverted_l_grid()
