from rob_agi.colored_grid import ColoredGrid

def solve_66e6c45b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 4x4 grid by moving the non-zero elements from the inner 2x2 square
    to the corners of the output grid, maintaining their relative positions.
    
    The transformation follows this pattern:
    - Top-left inner element moves to top-left corner
    - Top-right inner element moves to top-right corner
    - Bottom-left inner element moves to bottom-left corner
    - Bottom-right inner element moves to bottom-right corner
    """
    # Define the mapping of input positions to output positions
    mapping = {
        (1, 1): (0, 0),
        (1, 2): (0, 3),
        (2, 1): (3, 0),
        (2, 2): (3, 3)
    }
    
    # Initialize an empty 4x4 output grid
    output_values = [[0 for _ in range(4)] for _ in range(4)]
    
    # Move the elements from input positions to output positions
    for input_pos, output_pos in mapping.items():
        value = input_grid.get_cell(input_pos[0], input_pos[1])
        output_values[output_pos[0]][output_pos[1]] = value
    
    # Create and return the output ColoredGrid
    return ColoredGrid(values=output_values)
