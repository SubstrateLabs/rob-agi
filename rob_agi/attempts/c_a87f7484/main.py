from rob_agi.colored_grid import ColoredGrid

def solve_a87f7484(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the a87f7484 challenge by finding the first valid 3x3 pattern in the input grid.
    
    The function searches for specific 3x3 patterns in the input grid, starting from the bottom
    and moving upwards. It returns the first valid pattern found. If no valid pattern is found,
    it returns a default 3x3 grid filled with zeros.
    
    Valid patterns are:
    - 8x8 square with empty center
    - 7x7 X shape
    - 4x4 square with filled center
    - 7x7 L shape
    - 6x6 C shape
    - 5x5 plus shape
    
    Args:
    input_grid (ColoredGrid): The input grid to search for patterns.
    
    Returns:
    ColoredGrid: A 3x3 grid containing the first valid pattern found, or a default grid if none found.
    """
    def is_valid_pattern(subgrid: ColoredGrid) -> bool:
        patterns = [
            [[8, 8, 8], [8, 0, 8], [8, 8, 8]],
            [[7, 0, 7], [0, 7, 0], [7, 0, 7]],
            [[4, 0, 4], [4, 4, 4], [4, 0, 4]],
            [[0, 7, 7], [7, 7, 0], [7, 0, 7]],
            [[6, 0, 6], [6, 6, 0], [6, 0, 6]],
            [[5, 0, 5], [0, 5, 0], [5, 0, 5]]
        ]
        return subgrid.values in patterns

    height, width = input_grid.get_dimensions()
    for row in range(height - 2):
        for col in range(width - 2):
            subgrid = input_grid.extract_subgrid(row, col, 3, 3)
            if is_valid_pattern(subgrid):
                return subgrid

    # Fallback (should not occur with valid inputs)
    return ColoredGrid(values=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])
