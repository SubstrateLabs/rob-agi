from rob_agi.colored_grid import ColoredGrid

def solve_67a3c6ac(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 67a3c6ac challenge by performing a horizontal flip (mirror image) of the input grid.
    
    This function takes an input ColoredGrid and returns a new ColoredGrid that is the horizontal
    mirror image of the input. Each row of the grid is reversed, effectively flipping the entire
    grid from left to right.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: A new ColoredGrid object representing the horizontally flipped input grid.
    """
    return input_grid.flip_horizontal()
