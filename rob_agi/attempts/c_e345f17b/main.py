from rob_agi.colored_grid import ColoredGrid
import numpy as np
from typing import List, Tuple

def solve_e345f17b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an 8x4 input grid into a 4x4 output grid based on the distribution
    and patterns of magenta (6) and gray (5) squares.

    The algorithm works as follows:
    1. Creates a heat map of the input grid, giving higher weights to squares near corners and edges.
    2. Determines the number of yellow squares to place based on the total heat.
    3. Identifies potential positions for yellow squares based on influence scores.
    4. Places yellow squares ensuring no adjacency and considering the input pattern.
    5. Handles special cases like aligned yellow squares.

    Args:
    input_grid (ColoredGrid): An 8x4 grid representing the input pattern.

    Returns:
    ColoredGrid: A 4x4 grid representing the transformed output pattern.
    """
    input_array = np.array(input_grid.values)
    output = np.zeros((4, 4), dtype=int)

    def create_heat_map(grid: np.ndarray) -> np.ndarray:
        heat_map = np.zeros_like(grid, dtype=float)
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] in [5, 6]:
                    weight = 1.0
                    if grid[r, c] == 6:  # Give higher weight to magenta
                        weight = 1.5
                    # Give higher weight to corners and edges
                    if r in [0, rows-1] and c in [0, cols-1]:
                        weight *= 2
                    elif r in [0, rows-1] or c in [0, cols-1]:
                        weight *= 1.5
                    heat_map[r, c] = weight
        return heat_map

    heat_map = create_heat_map(input_array)
    total_heat = np.sum(heat_map)

    # Determine number of yellow squares
    num_yellow = 4 if total_heat > 20 else 3

    # Calculate influence scores for each position in the output grid
    influence_scores = np.zeros((4, 4))
    for r in range(4):
        for c in range(4):
            influence_scores[r, c] = np.sum(heat_map[r*2:(r+1)*2, c*2:(c+1)*2])

    # Place yellow squares
    yellow_positions = []
    for _ in range(num_yellow):
        max_score = np.max(influence_scores)
        if max_score == 0:
            break
        r, c = np.unravel_index(np.argmax(influence_scores), influence_scores.shape)
        yellow_positions.append((r, c))
        influence_scores[r, c] = 0
        # Set adjacent positions to 0 to avoid adjacency
        if r > 0: influence_scores[r-1, c] = 0
        if r < 3: influence_scores[r+1, c] = 0
        if c > 0: influence_scores[r, c-1] = 0
        if c < 3: influence_scores[r, c+1] = 0

    # Handle special case: align yellow squares if there's a strong horizontal or vertical pattern
    if num_yellow == 3:
        row_sums = np.sum(heat_map, axis=1)
        col_sums = np.sum(heat_map, axis=0)
        if np.max(row_sums) > 1.5 * np.mean(row_sums):
            yellow_positions = [(1, 0), (1, 1), (1, 2)]
        elif np.max(col_sums) > 1.5 * np.mean(col_sums):
            yellow_positions = [(0, 1), (1, 1), (2, 1)]

    # Place yellow squares in output grid
    for r, c in yellow_positions:
        output[r, c] = 4

    return ColoredGrid(values=output.tolist())
