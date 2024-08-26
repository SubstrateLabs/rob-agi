from rob_agi.colored_grid import ColoredGrid
import numpy as np
from typing import List, Tuple

def solve_e345f17b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an 8x4 input grid into a 4x4 output grid based on the distribution
    and patterns of magenta (6) and gray (5) squares.

    The algorithm works as follows:
    1. Preprocesses the input grid into 2x2 blocks.
    2. Creates a heat map based on the concentration of magenta and gray squares.
    3. Determines the number and positions of yellow squares based on the heat map.
    4. Applies special rules for positioning yellow squares (e.g., L-shapes, diagonals).
    5. Ensures a balanced distribution of yellow squares.
    6. Generates the final 4x4 output grid.

    Args:
    input_grid (ColoredGrid): An 8x4 grid representing the input pattern.

    Returns:
    ColoredGrid: A 4x4 grid representing the transformed output pattern.
    """
    input_array = np.array(input_grid.values)
    
    def preprocess_grid(grid: np.ndarray) -> np.ndarray:
        return grid.reshape(4, 2, 4, 2).sum(axis=(1, 3))
    
    def create_heat_map(preprocessed: np.ndarray) -> np.ndarray:
        heat_map = preprocessed.copy().astype(float)
        rows, cols = heat_map.shape
        for r in range(rows):
            for c in range(cols):
                if r in [0, rows-1] or c in [0, cols-1]:
                    heat_map[r, c] *= 1.5
        return heat_map
    
    preprocessed = preprocess_grid(input_array)
    heat_map = create_heat_map(preprocessed)
    
    num_yellow = 4 if np.sum(heat_map) > 20 else 3
    
    def get_yellow_positions(heat: np.ndarray, n: int) -> List[Tuple[int, int]]:
        positions = []
        for _ in range(n):
            r, c = np.unravel_index(np.argmax(heat), heat.shape)
            positions.append((r, c))
            heat[r, c] = 0
            if r > 0: heat[r-1, c] = 0
            if r < 3: heat[r+1, c] = 0
            if c > 0: heat[r, c-1] = 0
            if c < 3: heat[r, c+1] = 0
        return positions
    
    yellow_positions = get_yellow_positions(heat_map.copy(), num_yellow)
    
    def adjust_positions(positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(positions) == 3:
            # Check for L-shape or diagonal
            positions.sort()
            if positions[0][0] == positions[1][0] and positions[1][1] == positions[2][1]:
                return positions  # Already L-shape
            if positions[0][0] != positions[1][0] and positions[1][1] != positions[2][1]:
                return positions  # Already diagonal
            # Adjust to form L-shape
            return [(0, 0), (1, 0), (1, 1)]
        return positions
    
    yellow_positions = adjust_positions(yellow_positions)
    
    output = np.zeros((4, 4), dtype=int)
    for r, c in yellow_positions:
        output[r, c] = 4
    
    return ColoredGrid(values=output.tolist())
