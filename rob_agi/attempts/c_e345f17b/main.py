from rob_agi.colored_grid import ColoredGrid
import numpy as np

def solve_e345f17b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an 8x4 input grid into a 4x4 output grid based on the interaction
    between magenta (6) and gray (5) areas.

    The algorithm works as follows:
    1. Divides the input grid into two 4x4 halves (left and right).
    2. For each half, calculates the weighted centers of mass for magenta and gray areas.
    3. Determines the relative positions and strengths of magenta and gray concentrations.
    4. Places yellow (4) squares in the output grid to balance the color distributions.
    5. The placement of yellow squares follows these rules:
       - Strong concentrations of a color are balanced by yellows on the opposite side.
       - Balanced color distributions result in more centralized yellow placements.
       - Sparse color distributions may result in diagonal yellow placements.

    Args:
    input_grid (ColoredGrid): An 8x4 grid representing the input pattern.

    Returns:
    ColoredGrid: A 4x4 grid representing the transformed output pattern.
    """
    input_array = np.array(input_grid.values)
    magenta, gray, yellow = 6, 5, 4
    output = np.zeros((4, 4), dtype=int)

    def calculate_weighted_com(half_grid, color):
        positions = np.argwhere(half_grid == color)
        weights = np.sum(half_grid == color, axis=1)
        return np.average(positions, axis=0, weights=weights) if len(positions) > 0 else None

    def place_yellow(quadrant, strength, position):
        row, col = int(position[0] >= 2), int(position[1] >= 2)
        if strength > 0.6:
            output[2*row + (1-row), 2*quadrant + (1-col)] = yellow
        elif 0.3 < strength <= 0.6:
            output[2*row + (1-row), 2*quadrant + col] = yellow

    for half in range(2):
        half_grid = input_array[:, half*4:(half+1)*4]
        magenta_com = calculate_weighted_com(half_grid, magenta)
        gray_com = calculate_weighted_com(half_grid, gray)

        magenta_strength = np.sum(half_grid == magenta) / 16
        gray_strength = np.sum(half_grid == gray) / 16

        if magenta_com is not None and gray_com is not None:
            if abs(magenta_com[0] - gray_com[0]) > abs(magenta_com[1] - gray_com[1]):
                # Vertical separation
                place_yellow(half, magenta_strength, (3 - magenta_com[0], magenta_com[1]))
                place_yellow(half, gray_strength, (3 - gray_com[0], gray_com[1]))
            else:
                # Horizontal separation or mixed
                place_yellow(half, magenta_strength, (magenta_com[0], 3 - magenta_com[1]))
                place_yellow(half, gray_strength, (gray_com[0], 3 - gray_com[1]))
        elif magenta_com is not None:
            place_yellow(half, magenta_strength, (3 - magenta_com[0], 3 - magenta_com[1]))
        elif gray_com is not None:
            place_yellow(half, gray_strength, (3 - gray_com[0], 3 - gray_com[1]))

    # Ensure at least 3 yellow squares
    if np.sum(output == yellow) < 3:
        empty_positions = list(zip(*np.where(output == 0)))
        np.random.shuffle(empty_positions)
        for pos in empty_positions[:3 - np.sum(output == yellow)]:
            output[pos] = yellow

    return ColoredGrid(values=output.tolist())
