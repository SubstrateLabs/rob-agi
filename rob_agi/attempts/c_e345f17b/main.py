from rob_agi.colored_grid import ColoredGrid
import numpy as np

def solve_e345f17b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an 8x4 input grid into a 4x4 output grid based on the distribution
    of magenta (6) and gray (5) squares.

    The algorithm works as follows:
    1. Divides the 8x4 input grid into four 4x2 quadrants.
    2. Counts the number of magenta and gray squares in each quadrant.
    3. Determines the number of yellow squares to place (3 or 4) based on total colored squares.
    4. Analyzes horizontal and vertical color distributions.
    5. Places yellow squares in the 4x4 output grid to balance the input color distribution.
    6. Adjusts placement for even distributions and edge preferences.

    Args:
    input_grid (ColoredGrid): An 8x4 grid representing the input pattern.

    Returns:
    ColoredGrid: A 4x4 grid representing the transformed output pattern.
    """
    input_array = np.array(input_grid.values)
    magenta, gray, yellow = 6, 5, 4
    output = np.zeros((4, 4), dtype=int)

    # Count colored squares in each quadrant
    q1 = np.sum((input_array[:2, :4] == magenta) | (input_array[:2, :4] == gray))
    q2 = np.sum((input_array[:2, 4:] == magenta) | (input_array[:2, 4:] == gray))
    q3 = np.sum((input_array[2:, :4] == magenta) | (input_array[2:, :4] == gray))
    q4 = np.sum((input_array[2:, 4:] == magenta) | (input_array[2:, 4:] == gray))

    total_colored = q1 + q2 + q3 + q4
    num_yellow = 4 if total_colored > 16 else 3

    # Analyze horizontal and vertical distributions
    left_sum = q1 + q3
    right_sum = q2 + q4
    top_sum = q1 + q2
    bottom_sum = q3 + q4

    # Place yellow squares
    yellow_positions = []
    if left_sum > right_sum * 1.5:
        yellow_positions.extend([(1, 3), (2, 3)])
    elif right_sum > left_sum * 1.5:
        yellow_positions.extend([(1, 0), (2, 0)])
    
    if top_sum > bottom_sum * 1.5:
        yellow_positions.extend([(3, 1), (3, 2)])
    elif bottom_sum > top_sum * 1.5:
        yellow_positions.extend([(0, 1), (0, 2)])

    # Adjust for even distributions
    if not yellow_positions:
        yellow_positions = [(0, 0), (0, 3), (3, 0), (3, 3)]

    # Ensure correct number of yellow squares
    yellow_positions = yellow_positions[:num_yellow]
    while len(yellow_positions) < num_yellow:
        new_pos = (np.random.randint(4), np.random.randint(4))
        if new_pos not in yellow_positions:
            yellow_positions.append(new_pos)

    # Place yellow squares in output grid
    for pos in yellow_positions:
        output[pos] = yellow

    return ColoredGrid(values=output.tolist())
