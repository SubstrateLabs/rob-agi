from rob_agi.colored_grid import ColoredGrid

def solve_b1fc8b8e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 6x6 input grid into a 5x5 output grid based on the presence of sky blue (8) in corner regions.
    
    The function checks each 3x3 corner region in the input grid for the presence of sky blue (8).
    It then creates a 5x5 output grid where each corner is a 2x2 square that is sky blue (8) if the
    corresponding input corner contained at least one sky blue square. The center column and row
    are always black (0), creating a cross shape.
    
    Args:
    input_grid (ColoredGrid): A 6x6 input grid
    
    Returns:
    ColoredGrid: A 5x5 output grid with the described pattern
    """
    def check_corner_region(top: int, left: int) -> bool:
        for r in range(top, top + 3):
            for c in range(left, left + 3):
                if input_grid.get_cell(r, c) == 8:
                    return True
        return False

    output_grid = [[0 for _ in range(5)] for _ in range(5)]
    
    # Check and fill corners
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    output_corners = [(0, 0), (0, 3), (3, 0), (3, 3)]

    for (input_top, input_left), (output_top, output_left) in zip(corners, output_corners):
        if check_corner_region(input_top, input_left):
            for r in range(2):
                for c in range(2):
                    output_grid[output_top + r][output_left + c] = 8

    return ColoredGrid(values=output_grid)
