from rob_agi.colored_grid import ColoredGrid

def solve_b1fc8b8e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 6x6 input grid into a 5x5 output grid based on the presence of sky blue (8) in corner regions.
    
    The function checks each 3x3 corner region in the input grid for the presence of sky blue (8).
    It creates a 5x5 output grid where each corner is a 2x2 square that is sky blue (8) if the
    corresponding input corner contained at least two sky blue squares or if a specific trigger cell
    is sky blue. The center column and row are always black (0), creating a cross shape.
    
    Args:
    input_grid (ColoredGrid): A 6x6 input grid
    
    Returns:
    ColoredGrid: A 5x5 output grid with the described pattern
    """
    def create_empty_5x5_grid():
        return [[0 for _ in range(5)] for _ in range(5)]

    def count_blue_in_quadrant(top: int, left: int) -> int:
        return sum(input_grid.get_cell(r, c) == 8 
                   for r in range(top, top + 3) 
                   for c in range(left, left + 3))

    def check_trigger_cell(row: int, col: int) -> bool:
        return input_grid.get_cell(row, col) == 8

    output_grid = create_empty_5x5_grid()
    
    # Process each quadrant
    quadrants = [
        ((0, 0), (0, 0), (1, 0)),  # top-left
        ((0, 3), (0, 3), (0, 3)),  # top-right
        ((3, 0), (3, 0), (4, 1)),  # bottom-left
        ((3, 3), (3, 3), (5, 4))   # bottom-right
    ]

    for (input_top, input_left), (output_top, output_left), (trigger_row, trigger_col) in quadrants:
        if count_blue_in_quadrant(input_top, input_left) >= 2 or check_trigger_cell(trigger_row, trigger_col):
            for r in range(2):
                for c in range(2):
                    output_grid[output_top + r][output_left + c] = 8

    # Ensure the center cross is always black (0)
    for i in range(5):
        output_grid[2][i] = 0  # Center row
        output_grid[i][2] = 0  # Center column

    return ColoredGrid(values=output_grid)
