from rob_agi.colored_grid import ColoredGrid

def solve_2dee498d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 2dee498d challenge by extracting the repeating pattern from the input grid.
    
    The pattern observed is that the output is the smallest repeating subgrid
    that, when tiled horizontally, produces the input grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid containing the smallest repeating pattern.
    """
    height, width = input_grid.get_dimensions()
    
    for pattern_width in range(1, width + 1):
        if width % pattern_width == 0:
            pattern = input_grid.extract_subgrid(0, 0, height, pattern_width)
            tiled = pattern.tile_grid(width // pattern_width)
            if tiled == input_grid:
                return pattern
    
    # If no pattern is found, return the entire input grid
    return input_grid
