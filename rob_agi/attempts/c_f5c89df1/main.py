from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_f5c89df1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by identifying the core shape formed by sky blue (8) squares,
    and creating an idealized, symmetrical version of it.
    
    1. Identify all sky blue regions in the input grid.
    2. Determine the core shape and its bounding box.
    3. Create an expanded, idealized version of the shape with 4-fold rotational symmetry.
    4. Center the new shape vertically in the grid.
    5. Clean up any remaining non-sky blue or black squares.
    
    Returns a new grid with the transformed pattern.
    """
    # Step 1: Identify sky blue regions
    sky_blue_regions = input_grid.find_connected_regions(8)
    
    if not sky_blue_regions:
        return input_grid  # No transformation needed
    
    # Step 2: Determine core shape and bounding box
    all_sky_blue = [coord for region in sky_blue_regions for coord in region]
    min_r = min(r for r, _ in all_sky_blue)
    max_r = max(r for r, _ in all_sky_blue)
    min_c = min(c for _, c in all_sky_blue)
    max_c = max(c for _, c in all_sky_blue)
    
    # Step 3 & 4: Create and apply idealized shape
    output_grid = ColoredGrid(values=[[0 for _ in range(13)] for _ in range(13)])
    new_shape = create_idealized_shape(max_r - min_r + 1, max_c - min_c + 1)
    
    # Center the new shape vertically
    start_row = (13 - len(new_shape)) // 2
    start_col = (13 - len(new_shape[0])) // 2
    
    for r in range(len(new_shape)):
        for c in range(len(new_shape[0])):
            if new_shape[r][c] == 8:
                output_grid.set_cell(start_row + r, start_col + c, 8)
    
    return output_grid

def create_idealized_shape(height: int, width: int) -> List[List[int]]:
    """Create an idealized shape with 4-fold rotational symmetry."""
    size = max(height, width) + 2  # Add some padding
    shape = [[0 for _ in range(size)] for _ in range(size)]
    center = size // 2
    
    for r in range(size):
        for c in range(size):
            if (abs(r - center) <= 1 or abs(c - center) <= 1) and (r != center or c != center):
                shape[r][c] = 8
    
    # Add diagonal elements for more complex shapes
    shape[center-1][center-1] = 8
    shape[center-1][center+1] = 8
    shape[center+1][center-1] = 8
    shape[center+1][center+1] = 8
    
    return shape
