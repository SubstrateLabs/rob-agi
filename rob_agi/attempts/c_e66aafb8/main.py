from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List, Optional
import numpy as np

def solve_e66aafb8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e66aafb8 challenge by finding the most representative repeating pattern in the input grid.
    
    The function works as follows:
    1. Preprocess the input grid to identify non-black areas.
    2. Iterate through potential pattern sizes and aspect ratios.
    3. For each size, find patterns that repeat at least twice in the grid.
    4. Clean and refine the patterns by removing black cells and unnecessary repetitions.
    5. Select the best pattern based on size and coverage of non-black areas.
    6. Handle edge cases and validate the final output.
    
    Returns:
        ColoredGrid: A new grid containing the extracted pattern.
    """
    rows, cols = input_grid.get_dimensions()
    grid = np.array([[input_grid.get_cell(r, c) for c in range(cols)] for r in range(rows)])
    non_black_mask = grid != 0
    
    if not np.any(non_black_mask):
        return ColoredGrid(values=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    def check_pattern_repetition(pattern: np.ndarray) -> Tuple[bool, int]:
        p_rows, p_cols = pattern.shape
        repetitions = 0
        for r in range(0, rows - p_rows + 1, p_rows):
            for c in range(0, cols - p_cols + 1, p_cols):
                if np.array_equal(grid[r:r+p_rows, c:c+p_cols], pattern):
                    repetitions += 1
        return repetitions >= 2, repetitions
    
    def clean_pattern(pattern: np.ndarray) -> np.ndarray:
        pattern = pattern[~np.all(pattern == 0, axis=1)]
        pattern = pattern[:, ~np.all(pattern == 0, axis=0)]
        return pattern
    
    best_pattern = None
    best_pattern_size = 0
    
    for size in range(min(12, rows, cols), 1, -1):
        for aspect_ratio in range(1, 4):
            for height, width in [(size, size//aspect_ratio), (size//aspect_ratio, size)]:
                if height * width > 64 or height < 2 or width < 2:  # Max size 8x8, min size 2x2
                    continue
                
                for r in range(rows - height + 1):
                    for c in range(cols - width + 1):
                        pattern = grid[r:r+height, c:c+width]
                        repeats, _ = check_pattern_repetition(pattern)
                        
                        if repeats:
                            cleaned_pattern = clean_pattern(pattern)
                            if cleaned_pattern.size > best_pattern_size:
                                best_pattern = cleaned_pattern
                                best_pattern_size = cleaned_pattern.size
    
    if best_pattern is None:
        # Fallback: return the largest non-black rectangular area up to 8x5
        non_black_rows = np.any(non_black_mask, axis=1)
        non_black_cols = np.any(non_black_mask, axis=0)
        height = min(8, np.sum(non_black_rows))
        width = min(5, np.sum(non_black_cols))
        best_pattern = grid[non_black_rows][:height, :][:, non_black_cols][:, :width]
    
    return ColoredGrid(values=best_pattern.tolist())
