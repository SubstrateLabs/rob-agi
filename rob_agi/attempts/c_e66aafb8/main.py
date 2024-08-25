from rob_agi.colored_grid import ColoredGrid
import numpy as np
from scipy import signal
from typing import Tuple, List

def solve_e66aafb8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e66aafb8 challenge by finding the most representative subgrid pattern in the input grid.
    
    The function works as follows:
    1. Preprocess the input grid to create a numpy array and calculate overall color distribution.
    2. Define a scoring function for subgrids based on color diversity, representativeness, and pattern complexity.
    3. Generate candidate subgrids of various sizes (2x2 up to 8x8 or half the input size).
    4. Score and rank subgrids, maintaining a list of top candidates.
    5. Refine top candidates by adjusting boundaries slightly.
    6. Select the best subgrid after refinement.
    7. Post-process the selected subgrid to handle edge cases and ensure minimum size.
    8. Return the result as a ColoredGrid object.
    
    Returns:
        ColoredGrid: A new grid containing the extracted representative pattern.
    """
    rows, cols = input_grid.get_dimensions()
    grid = np.array([[input_grid.get_cell(r, c) for c in range(cols)] for r in range(rows)])
    non_black_mask = grid != 0
    
    if not np.any(non_black_mask):
        return ColoredGrid(values=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    overall_distribution = np.bincount(grid[non_black_mask].flatten(), minlength=10)[1:] / np.sum(non_black_mask)
    
    def score_subgrid(subgrid: np.ndarray) -> float:
        if np.all(subgrid == 0):
            return -np.inf
        
        unique_colors, color_counts = np.unique(subgrid, return_counts=True)
        color_diversity = len(unique_colors)
        color_distribution = color_counts / np.sum(color_counts)
        
        representativeness = 1 - np.sum(np.abs(color_distribution - overall_distribution[:len(color_distribution)]))
        
        edges = np.abs(signal.convolve2d(subgrid, np.array([[1, -1], [-1, 1]]), mode='valid'))
        pattern_complexity = np.sum(edges) / subgrid.size
        
        black_penalty = -0.5 * np.sum(subgrid == 0) / subgrid.size
        
        score = (color_diversity + representativeness + pattern_complexity + black_penalty) / np.log(subgrid.size)
        return score
    
    def generate_candidates() -> List[Tuple[np.ndarray, float]]:
        candidates = []
        max_size = min(8, min(rows, cols) // 2)
        for height in range(2, max_size + 1):
            for width in range(2, max_size + 1):
                for r in range(rows - height + 1):
                    for c in range(cols - width + 1):
                        subgrid = grid[r:r+height, c:c+width]
                        score = score_subgrid(subgrid)
                        candidates.append((subgrid, score))
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:10]
    
    def refine_subgrid(subgrid: np.ndarray) -> np.ndarray:
        height, width = subgrid.shape
        best_refined = subgrid
        best_score = score_subgrid(subgrid)
        
        for dh in [-1, 0, 1]:
            for dw in [-1, 0, 1]:
                if dh == 0 and dw == 0:
                    continue
                new_height, new_width = height + dh, width + dw
                if new_height < 2 or new_width < 2:
                    continue
                r, c = np.unravel_index(np.argmax(grid[:rows-new_height+1, :cols-new_width+1]), (rows-new_height+1, cols-new_width+1))
                refined = grid[r:r+new_height, c:c+new_width]
                score = score_subgrid(refined)
                if score > best_score:
                    best_refined = refined
                    best_score = score
        
        return best_refined
    
    candidates = generate_candidates()
    best_subgrid = max((refine_subgrid(subgrid) for subgrid, _ in candidates), key=score_subgrid)
    
    # Post-process: remove black edges if possible
    while np.any(best_subgrid[0] == 0) and best_subgrid.shape[0] > 2:
        best_subgrid = best_subgrid[1:]
    while np.any(best_subgrid[-1] == 0) and best_subgrid.shape[0] > 2:
        best_subgrid = best_subgrid[:-1]
    while np.any(best_subgrid[:, 0] == 0) and best_subgrid.shape[1] > 2:
        best_subgrid = best_subgrid[:, 1:]
    while np.any(best_subgrid[:, -1] == 0) and best_subgrid.shape[1] > 2:
        best_subgrid = best_subgrid[:, :-1]
    
    # Ensure minimum size of 2x2
    if best_subgrid.shape[0] < 2 or best_subgrid.shape[1] < 2:
        best_subgrid = np.pad(best_subgrid, ((0, max(0, 2 - best_subgrid.shape[0])), (0, max(0, 2 - best_subgrid.shape[1]))), mode='edge')
    
    return ColoredGrid(values=best_subgrid.tolist())
