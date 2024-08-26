from rob_agi.colored_grid import ColoredGrid
import numpy as np

def solve_e345f17b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an 8x4 input grid into a 4x4 output grid based on the interaction
    between magenta (6) and gray (5) areas.

    The algorithm works as follows:
    1. Divides the input grid into two 4x4 halves (left and right).
    2. For each half, calculates the centers of mass for magenta and gray areas.
    3. Determines the relative positions of magenta and gray concentrations.
    4. Places yellow (4) squares in the output grid based on the interaction between
       magenta and gray areas in each half.
    5. The placement of yellow squares follows these rules:
       - If magenta and gray are on opposite sides vertically, place yellows horizontally.
       - If magenta and gray are on the same side vertically, place yellows vertically on the opposite side.
       - If one color dominates or colors are mixed, place yellows diagonally.

    Args:
    input_grid (ColoredGrid): An 8x4 grid representing the input pattern.

    Returns:
    ColoredGrid: A 4x4 grid representing the transformed output pattern.
    """
    # Convert input grid to numpy array for easier manipulation
    input_array = np.array(input_grid.values)

    # Define colors
    magenta, gray = 6, 5

    # Initialize output grid
    output = np.zeros((4, 4), dtype=int)

    # Process each half
    for half in range(2):
        half_grid = input_array[:, half*4:(half+1)*4]
        
        # Calculate centers of mass for magenta and gray
        magenta_com = np.mean(np.argwhere(half_grid == magenta), axis=0) if np.any(half_grid == magenta) else None
        gray_com = np.mean(np.argwhere(half_grid == gray), axis=0) if np.any(half_grid == gray) else None

        if magenta_com is not None and gray_com is not None:
            # Determine relative positions
            magenta_top = magenta_com[0] < 2
            gray_top = gray_com[0] < 2

            if magenta_top != gray_top:
                # Colors on opposite sides vertically, place yellows horizontally
                output[2, half*2:half*2+2] = 4
            elif (np.sum(half_grid == magenta) > 3 and np.sum(half_grid == gray) > 3) or \
                 (np.sum(half_grid == magenta) <= 1 and np.sum(half_grid == gray) <= 1):
                # Both colors dominant or both sparse, place yellows diagonally
                output[half, half*2] = 4
                output[3-half, half*2+1] = 4
            else:
                # Colors on same side, place yellows vertically on opposite side
                output[0, half*2+1] = 4
                output[3, half*2+1] = 4

    return ColoredGrid(values=output.tolist())
