from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_929ab4e9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by:
    1. Identifying the masked area (red squares)
    2. Analyzing the global pattern and symmetry
    3. Reconstructing the pattern in the masked area
    4. Verifying and adjusting for perfect symmetry
    5. Returning the modified grid with the reconstructed pattern
    """
    masked_area = identify_masked_area(input_grid)
    symmetry_axes = find_symmetry_axes(input_grid)
    reconstructed_grid = reconstruct_pattern(input_grid, masked_area, symmetry_axes)
    return verify_and_adjust_symmetry(reconstructed_grid, symmetry_axes)

def identify_masked_area(grid: ColoredGrid) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == 2]

def find_symmetry_axes(grid: ColoredGrid) -> Tuple[bool, bool, bool, bool]:
    horizontal = all(grid.values[r] == grid.values[-r-1] for r in range(grid.num_rows // 2))
    vertical = all(row[c] == row[-c-1] for row in grid.values for c in range(grid.num_cols // 2))
    diagonal1 = all(grid.values[r][c] == grid.values[c][r] for r in range(grid.num_rows) for c in range(grid.num_cols))
    diagonal2 = all(grid.values[r][c] == grid.values[grid.num_rows-1-c][grid.num_cols-1-r] 
                    for r in range(grid.num_rows) for c in range(grid.num_cols))
    return horizontal, vertical, diagonal1, diagonal2

def reconstruct_pattern(grid: ColoredGrid, masked_area: List[Tuple[int, int]], 
                        symmetry_axes: Tuple[bool, bool, bool, bool]) -> ColoredGrid:
    output_grid = grid.deep_copy()
    horizontal, vertical, diagonal1, diagonal2 = symmetry_axes
    
    for r, c in masked_area:
        candidates = []
        if horizontal:
            candidates.append(grid.values[grid.num_rows-1-r][c])
        if vertical:
            candidates.append(grid.values[r][grid.num_cols-1-c])
        if diagonal1:
            candidates.append(grid.values[c][r])
        if diagonal2:
            candidates.append(grid.values[grid.num_rows-1-c][grid.num_cols-1-r])
        
        if candidates:
            output_grid.values[r][c] = max(set(candidates), key=candidates.count)
        else:
            # If no symmetry, use the most common color in the grid
            colors = [grid.values[i][j] for i in range(grid.num_rows) for j in range(grid.num_cols) if grid.values[i][j] != 2]
            output_grid.values[r][c] = max(set(colors), key=colors.count)
    
    return output_grid

def verify_and_adjust_symmetry(grid: ColoredGrid, symmetry_axes: Tuple[bool, bool, bool, bool]) -> ColoredGrid:
    horizontal, vertical, diagonal1, diagonal2 = symmetry_axes
    adjusted_grid = grid.deep_copy()
    
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            symmetry_colors = [adjusted_grid.values[r][c]]
            if horizontal:
                symmetry_colors.append(adjusted_grid.values[grid.num_rows-1-r][c])
            if vertical:
                symmetry_colors.append(adjusted_grid.values[r][grid.num_cols-1-c])
            if diagonal1:
                symmetry_colors.append(adjusted_grid.values[c][r])
            if diagonal2:
                symmetry_colors.append(adjusted_grid.values[grid.num_rows-1-c][grid.num_cols-1-r])
            
            final_color = max(set(symmetry_colors), key=symmetry_colors.count)
            adjusted_grid.values[r][c] = final_color
            
            if horizontal:
                adjusted_grid.values[grid.num_rows-1-r][c] = final_color
            if vertical:
                adjusted_grid.values[r][grid.num_cols-1-c] = final_color
            if diagonal1:
                adjusted_grid.values[c][r] = final_color
            if diagonal2:
                adjusted_grid.values[grid.num_rows-1-c][grid.num_cols-1-r] = final_color
    
    return adjusted_grid
