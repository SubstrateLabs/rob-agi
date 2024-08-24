from rob_agi.colored_grid import ColoredGrid

def solve_00576224(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 2x2 input grid into a 6x6 output grid by repeating and alternating the elements.
    
    The pattern is as follows:
    - Rows 1-2: Repeat the input grid horizontally three times.
    - Rows 3-4: Swap the columns of the input grid and repeat horizontally three times.
    - Rows 5-6: Identical to rows 1-2.
    
    Args:
    input_grid (ColoredGrid): A 2x2 input grid

    Returns:
    ColoredGrid: A 6x6 output grid following the described pattern
    """
    # Extract the 2x2 input values
    a, b = input_grid.values[0]
    c, d = input_grid.values[1]
    
    # Create the 6x6 output grid
    output_values = [
        [a, b, a, b, a, b],
        [c, d, c, d, c, d],
        [b, a, b, a, b, a],
        [d, c, d, c, d, c],
        [a, b, a, b, a, b],
        [c, d, c, d, c, d]
    ]
    
    return ColoredGrid(values=output_values)
