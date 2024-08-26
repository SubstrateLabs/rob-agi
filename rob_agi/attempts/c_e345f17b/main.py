from rob_agi.colored_grid import ColoredGrid
import numpy as np
from typing import List, Tuple

def solve_e345f17b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an 8x4 input grid into a 4x4 output grid based on the distribution
    and patterns of magenta (6) and gray (5) squares.

    The algorithm works as follows:
    1. Analyzes the input grid's color distribution in four quadrants.
    2. Identifies connected components of magenta and gray squares in each quadrant.
    3. Determines the number of yellow squares to place based on total colored squares.
    4. Places yellow squares in the output grid corners based on input patterns.
    5. Adjusts yellow square placement to ensure no adjacency and correct count.

    Args:
    input_grid (ColoredGrid): An 8x4 grid representing the input pattern.

    Returns:
    ColoredGrid: A 4x4 grid representing the transformed output pattern.
    """
    input_array = np.array(input_grid.values)
    output = np.zeros((4, 4), dtype=int)

    def get_connected_components(quadrant: np.ndarray) -> List[int]:
        visited = set()
        components = []
        for r in range(quadrant.shape[0]):
            for c in range(quadrant.shape[1]):
                if (r, c) not in visited and quadrant[r, c] in [5, 6]:
                    component = []
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) not in visited:
                            visited.add((cr, cc))
                            component.append(quadrant[cr, cc])
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < quadrant.shape[0] and 0 <= nc < quadrant.shape[1] and quadrant[nr, nc] in [5, 6]:
                                    stack.append((nr, nc))
                    components.append(len(component))
        return components

    # Analyze input quadrants
    quadrants = [
        input_array[:2, :4], input_array[:2, 4:],
        input_array[2:, :4], input_array[2:, 4:]
    ]
    quadrant_scores = [sum(get_connected_components(q)) for q in quadrants]

    # Determine number of yellow squares
    total_colored = sum(quadrant_scores)
    num_yellow = 4 if total_colored > 16 else 3

    # Place yellow squares based on quadrant scores
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    yellow_positions = []
    for _ in range(num_yellow):
        max_score_index = quadrant_scores.index(max(quadrant_scores))
        yellow_positions.append(corners[max_score_index])
        quadrant_scores[max_score_index] = -1  # Mark as used

    # Adjust yellow positions to avoid adjacency
    def are_adjacent(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    for i in range(len(yellow_positions)):
        for j in range(i + 1, len(yellow_positions)):
            if are_adjacent(yellow_positions[i], yellow_positions[j]):
                # Move one of the yellow squares to the center if adjacent
                yellow_positions[j] = (1, 1)
                break

    # Place yellow squares in output grid
    for pos in yellow_positions:
        output[pos] = 4

    return ColoredGrid(values=output.tolist())
