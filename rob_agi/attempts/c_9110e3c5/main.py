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
    
    The function analyzes the framing pattern of non-black cells in the input grid,
    particularly focusing on the edges and corners. Based on this analysis,
    it returns one of three predefined 3x3 patterns:
    1. Backwards "C" pattern (right-bottom frame)
    2. Inverted "L" pattern (left-bottom frame)
    3. Horizontal stripe pattern (uniform frame or low non-black cell count)

    Args:
    input_grid (ColoredGrid): A 7x7 ColoredGrid object representing the input.

    Returns:
    ColoredGrid: A 3x3 ColoredGrid object representing the output pattern.
    """
    rows, cols = input_grid.get_dimensions()
    total_non_black = 0
    left_edge = 0
    right_edge = 0
    top_edge = 0
    bottom_edge = 0
    corners = 0

    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) != 0:  # Non-black cell
                total_non_black += 1
                if c == 0:
                    left_edge += 1
                if c == cols - 1:
                    right_edge += 1
                if r == 0:
                    top_edge += 1
                if r == rows - 1:
                    bottom_edge += 1
                if (r, c) in [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]:
                    corners += 1

    if total_non_black < 5:
        return create_horizontal_stripe_grid()

    if corners >= 3:
        return create_backwards_c_grid()

    left_frame_score = left_edge + bottom_edge
    right_frame_score = right_edge + bottom_edge
    uniform_frame_score = (left_edge + right_edge + top_edge + bottom_edge) / 4
    threshold = 1.5

    if right_frame_score > left_frame_score and right_frame_score > uniform_frame_score + threshold:
        return create_backwards_c_grid()
    elif left_frame_score > right_frame_score and left_frame_score > uniform_frame_score + threshold:
        return create_inverted_l_grid()
    else:
        return create_horizontal_stripe_grid()
