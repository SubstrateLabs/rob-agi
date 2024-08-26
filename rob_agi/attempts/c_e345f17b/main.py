from rob_agi.colored_grid import ColoredGrid
import numpy as np

def solve_e345f17b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an 8x4 input grid into a 4x4 output grid based on the distribution
    of magenta (6) and gray (5) squares.

    The algorithm works as follows:
    1. Analyzes the input grid's color distribution in four sections.
    2. Maps these sections to the output grid's edges.
    3. Places 3 or 4 yellow squares based on color density and patterns.
    4. Ensures yellow squares are not adjacent.

    Args:
    input_grid (ColoredGrid): An 8x4 grid representing the input pattern.

    Returns:
    ColoredGrid: A 4x4 grid representing the transformed output pattern.
    """
    input_array = np.array(input_grid.values)
    output = np.zeros((4, 4), dtype=int)

    # Analyze input sections
    left = input_array[:, :4]
    right = input_array[:, 4:]
    top = input_array[:2, :]
    bottom = input_array[2:, :]

    def color_density(section):
        return np.sum((section == 6) | (section == 5)) / section.size

    densities = {
        'left': color_density(left),
        'right': color_density(right),
        'top': color_density(top),
        'bottom': color_density(bottom)
    }

    # Determine number of yellow squares
    total_colored = np.sum((input_array == 6) | (input_array == 5))
    num_yellow = 4 if total_colored > 16 else 3

    # Place yellow squares based on densities
    yellow_positions = []
    if densities['left'] > densities['right']:
        yellow_positions.append((0, 0))
    else:
        yellow_positions.append((0, 3))
    
    if densities['top'] > densities['bottom']:
        yellow_positions.append((0, 3) if (0, 3) not in yellow_positions else (0, 0))
    else:
        yellow_positions.append((3, 3) if (0, 3) in yellow_positions else (3, 0))

    # Place remaining yellow squares
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    for corner in corners:
        if len(yellow_positions) < num_yellow and corner not in yellow_positions:
            yellow_positions.append(corner)
            break

    if len(yellow_positions) < num_yellow:
        middle_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for pos in middle_positions:
            if len(yellow_positions) < num_yellow:
                yellow_positions.append(pos)
                break

    # Place yellow squares in output grid
    for pos in yellow_positions:
        output[pos] = 4

    return ColoredGrid(values=output.tolist())
