from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_f5c89df1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by identifying the core shape formed by sky blue (8) squares,
    and creating an idealized, symmetrical version of it.
    
    1. Identify all sky blue regions in the input grid.
    2. Determine the core shape and its center.
    3. Create an expanded, idealized version of the shape.
    4. Apply the transformation to the grid.
    5. Clean up any remaining non-sky blue or black squares.
    
    Returns a new grid with the transformed pattern.
    """
    # Step 1: Identify sky blue regions
    sky_blue_regions = input_grid.find_connected_regions(8)
    
    if not sky_blue_regions:
        return input_grid  # No transformation needed
    
    # Step 2: Determine core shape and center
    all_sky_blue = [coord for region in sky_blue_regions for coord in region]
    center = find_center(all_sky_blue)
    
    # Step 3 & 4: Create and apply idealized shape
    output_grid = input_grid.deep_copy()
    output_grid = create_idealized_shape(output_grid, center, all_sky_blue)
    
    # Step 5: Clean up
    rows, cols = output_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) not in [0, 8]:
                output_grid.set_cell(r, c, 0)
    
    return output_grid

def find_center(coordinates: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Find the center of a set of coordinates."""
    x_sum = sum(x for x, _ in coordinates)
    y_sum = sum(y for _, y in coordinates)
    count = len(coordinates)
    return (x_sum // count, y_sum // count)

def create_idealized_shape(grid: ColoredGrid, center: Tuple[int, int], sky_blue_coords: List[Tuple[int, int]]) -> ColoredGrid:
    """Create an idealized shape based on the input pattern."""
    rows, cols = grid.get_dimensions()
    max_distance = max(max(abs(r - center[0]), abs(c - center[1])) for r, c in sky_blue_coords)
    
    # Create a larger, symmetrical shape
    for r in range(rows):
        for c in range(cols):
            distance = max(abs(r - center[0]), abs(c - center[1]))
            if distance <= max_distance + 1:
                if distance == max_distance + 1 or (r - center[0]) % 2 == 0 or (c - center[1]) % 2 == 0:
                    grid.set_cell(r, c, 8)
                else:
                    grid.set_cell(r, c, 0)
            else:
                grid.set_cell(r, c, 0)
    
    return grid
