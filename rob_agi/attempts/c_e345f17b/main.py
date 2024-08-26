from rob_agi.colored_grid import ColoredGrid
import numpy as np

def solve_e345f17b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an 8x4 input grid into a 4x4 output grid based on the interaction
    between magenta (6) and gray (5) areas.

    The algorithm works as follows:
    1. Divides the input grid into four 4x4 quadrants.
    2. For each quadrant, calculates the centers of mass for magenta and gray areas.
    3. Computes an interaction score for each cell in the output grid based on
       the proximity to magenta and gray centers and their concentrations.
    4. Places yellow (4) squares in the output grid where the interaction score
       exceeds a certain threshold, and black (0) squares elsewhere.

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

    # Process each quadrant
    for i in range(2):
        for j in range(4):
            quadrant = input_array[i*2:(i+1)*2, j:j+2]
            
            # Calculate centers of mass for magenta and gray
            magenta_com = np.mean(np.argwhere(quadrant == magenta), axis=0) if np.any(quadrant == magenta) else None
            gray_com = np.mean(np.argwhere(quadrant == gray), axis=0) if np.any(quadrant == gray) else None

            # Calculate interaction score
            score = 0
            if magenta_com is not None and gray_com is not None:
                distance = np.linalg.norm(magenta_com - gray_com)
                magenta_count = np.sum(quadrant == magenta)
                gray_count = np.sum(quadrant == gray)
                score = (magenta_count * gray_count) / (distance + 1)  # Add 1 to avoid division by zero

            # Set output cell based on score
            output[i, j] = 4 if score > 1 else 0  # Threshold of 1 seems to work well

    return ColoredGrid(values=output.tolist())
