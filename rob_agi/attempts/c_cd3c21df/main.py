from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List, Dict

def solve_cd3c21df(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Find the unique, largest subgrid pattern in the input grid.
    
    This function identifies the largest subgrid that appears only once in the input grid.
    If multiple such subgrids exist, it returns the one that appears first (top-left to bottom-right).
    
    Steps:
    1. Generate all possible subgrids
    2. Count occurrences of each unique subgrid
    3. Find the largest subgrid(s) that appear only once
    4. Return the first largest unique subgrid
    
    Args:
    input_grid (ColoredGrid): The input grid to analyze

    Returns:
    ColoredGrid: The unique largest subgrid found, or None if no unique subgrid exists
    """
    rows, cols = input_grid.get_dimensions()
    
    def subgrid_to_tuple(subgrid: ColoredGrid) -> Tuple[Tuple[int, ...]]:
        return tuple(tuple(row) for row in subgrid.values)
    
    def generate_subgrids() -> List[Tuple[int, int, int, int, ColoredGrid]]:
        subgrids = []
        for top in range(rows):
            for left in range(cols):
                for height in range(1, rows - top + 1):
                    for width in range(1, cols - left + 1):
                        subgrid = input_grid.extract_subgrid(top, left, height, width)
                        subgrids.append((top, left, height, width, subgrid))
        return subgrids
    
    subgrids = generate_subgrids()
    subgrid_counts: Dict[Tuple[Tuple[int, ...]], List[Tuple[int, int, int, int, ColoredGrid]]] = {}
    
    for top, left, height, width, subgrid in subgrids:
        key = subgrid_to_tuple(subgrid)
        if key not in subgrid_counts:
            subgrid_counts[key] = []
        subgrid_counts[key].append((top, left, height, width, subgrid))
    
    unique_subgrids = [subgrids[0] for subgrids in subgrid_counts.values() if len(subgrids) == 1]
    
    if not unique_subgrids:
        return None
    
    largest_subgrids = [subgrid for subgrid in unique_subgrids if subgrid[2] * subgrid[3] == max(s[2] * s[3] for s in unique_subgrids)]
    
    largest_subgrids.sort(key=lambda x: (x[0], x[1]))  # Sort by top, then left
    
    return largest_subgrids[0][4]  # Return the ColoredGrid object
