from rob_agi.colored_grid import ColoredGrid
import numpy as np
from typing import Tuple

def solve_e66aafb8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e66aafb8 challenge by finding the most representative subgrid pattern in the input grid.
    
    The function works as follows:
    1. Preprocess the input grid to create a numpy array and identify non-black areas.
    2. Define a scoring function for subgrids based on color diversity and representativeness.
    3. Iterate through potential subgrid sizes from 3x3 up to 8x8.
    4. For each size, slide a window across the grid and score each subgrid.
    5. Select the highest-scoring subgrid that doesn't contain black cells.
    6. If no satisfactory subgrid is found, use a fallback method to extract the largest non-black area.
    7. Post-process the selected subgrid to remove any all-black rows or columns.
    8. Return the result as a ColoredGrid object.
    
    Returns:
        ColoredGrid: A new grid containing the extracted representative pattern.
    """
    rows, cols = input_grid.get_dimensions()
    grid = np.array([[input_grid.get_cell(r, c) for c in range(cols)] for r in range(rows)])
    non_black_mask = grid != 0
    
    if not np.any(non_black_mask):
        return ColoredGrid(values=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    def score_subgrid(subgrid: np.ndarray) -> float:
        if np.any(subgrid == 0):
            return -1  # Reject subgrids with black cells
        
        unique_colors, color_counts = np.unique(subgrid, return_counts=True)
        color_diversity = len(unique_colors)
        color_distribution = color_counts / np.sum(color_counts)
        
        overall_distribution = np.bincount(grid[non_black_mask].flatten(), minlength=10)[1:] / np.sum(non_black_mask)
        representativeness = 1 - np.sum(np.abs(color_distribution - overall_distribution[:len(color_distribution)]))
        
        return color_diversity + representativeness
    
    best_subgrid = None
    best_score = -1
    
    for size in range(3, 9):
        for aspect_ratio in range(1, 4):
            for height, width in [(size, size//aspect_ratio), (size//aspect_ratio, size)]:
                if height * width > 64 or height < 2 or width < 2 or height > rows or width > cols:
                    continue
                
                for r in range(rows - height + 1):
                    for c in range(cols - width + 1):
                        subgrid = grid[r:r+height, c:c+width]
                        score = score_subgrid(subgrid)
                        if score > best_score:
                            best_subgrid = subgrid
                            best_score = score
    
    if best_subgrid is None:
        # Fallback: extract the largest contiguous non-black area
        from scipy.ndimage import label
        labeled, num_features = label(non_black_mask)
        largest_area = max([(labeled == i).sum() for i in range(1, num_features + 1)])
        largest_label = [(labeled == i).sum() for i in range(1, num_features + 1)].index(largest_area) + 1
        largest_region = labeled == largest_label
        r_indices, c_indices = np.where(largest_region)
        r_min, r_max = r_indices.min(), r_indices.max()
        c_min, c_max = c_indices.min(), c_indices.max()
        best_subgrid = grid[r_min:r_max+1, c_min:c_max+1]
        best_subgrid = best_subgrid[:min(8, best_subgrid.shape[0]), :min(8, best_subgrid.shape[1])]
    
    # Post-process: remove any all-black rows or columns
    best_subgrid = best_subgrid[~np.all(best_subgrid == 0, axis=1)]
    best_subgrid = best_subgrid[:, ~np.all(best_subgrid == 0, axis=0)]
    
    # Ensure minimum size of 2x2
    if best_subgrid.shape[0] < 2 or best_subgrid.shape[1] < 2:
        best_subgrid = np.pad(best_subgrid, ((0, max(0, 2 - best_subgrid.shape[0])), (0, max(0, 2 - best_subgrid.shape[1]))), mode='edge')
    
    return ColoredGrid(values=best_subgrid.tolist())
