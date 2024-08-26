from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_f5c89df1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by identifying the core shape formed by sky blue (8) squares,
    and creating an idealized, symmetrical version of it.
    
    1. Identify all sky blue regions in the input grid.
    2. Create an abstract representation of the input shape.
    3. Design a new symmetrical shape inspired by the input.
    4. Size and position the new shape within the grid.
    5. Create the output grid with the new shape.
    
    Returns a new grid with the transformed pattern.
    """
    # Step 1: Identify sky blue regions
    sky_blue_regions = input_grid.find_connected_regions(8)
    
    if not sky_blue_regions:
        return input_grid  # No transformation needed
    
    # Step 2: Create abstract representation
    all_sky_blue = set(coord for region in sky_blue_regions for coord in region)
    centroid = calculate_centroid(all_sky_blue)
    normalized_shape = normalize_shape(all_sky_blue, centroid)
    
    # Step 3 & 4: Design and size new shape
    new_shape = design_symmetrical_shape(normalized_shape)
    
    # Step 5: Create output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(13)] for _ in range(13)])
    place_shape_on_grid(output_grid, new_shape, centroid)
    
    return output_grid

def calculate_centroid(coords: Set[Tuple[int, int]]) -> Tuple[float, float]:
    """Calculate the centroid of a set of coordinates."""
    if not coords:
        return (0, 0)
    return (sum(r for r, _ in coords) / len(coords),
            sum(c for _, c in coords) / len(coords))

def normalize_shape(coords: Set[Tuple[int, int]], centroid: Tuple[float, float]) -> Set[Tuple[int, int]]:
    """Normalize coordinates relative to the centroid."""
    cr, cc = centroid
    return {(int(r - cr), int(c - cc)) for r, c in coords}

def design_symmetrical_shape(normalized_shape: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Design a new symmetrical shape inspired by the input."""
    max_extent = max(max(abs(r), abs(c)) for r, c in normalized_shape)
    size = min(max(max_extent * 2, 3), 5)  # Ensure size is between 3 and 5
    
    new_shape = set()
    for r in range(-size, size + 1):
        for c in range(-size, size + 1):
            if (abs(r) == size or abs(c) == size or
                abs(r) + abs(c) == size or
                (r == 0 and abs(c) <= size - 1) or
                (c == 0 and abs(r) <= size - 1)):
                new_shape.add((r, c))
    
    return new_shape

def place_shape_on_grid(grid: ColoredGrid, shape: Set[Tuple[int, int]], centroid: Tuple[float, float]):
    """Place the new shape on the grid, centered around the original centroid."""
    cr, cc = centroid
    center_r, center_c = int(cr), int(cc)
    
    for r, c in shape:
        grid_r, grid_c = center_r + r, center_c + c
        if 0 <= grid_r < 13 and 0 <= grid_c < 13:
            grid.set_cell(grid_r, grid_c, 8)
