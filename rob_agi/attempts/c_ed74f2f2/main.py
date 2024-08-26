from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ed74f2f2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 9x5 input grid into a 3x3 output grid based on the following rules:
    1. Creates a density grid by analyzing 3x3 sections of the input grid.
    2. Determines the output color based on the density and distribution of gray cells.
    3. Creates an initial shape based on the density grid and chosen color.
    4. Refines the shape to better reflect input characteristics.
    5. Makes final adjustments to ensure a valid and interesting output.
    6. Returns the final 3x3 ColoredGrid output.
    """
    density_grid = create_density_grid(input_grid)
    color = determine_color(density_grid)
    initial_shape = create_initial_shape(density_grid, color)
    refined_shape = refine_shape(initial_shape, density_grid)
    final_shape = final_adjustments(refined_shape)
    return ColoredGrid(values=final_shape)

def create_density_grid(input_grid: ColoredGrid) -> List[List[int]]:
    density = [[0 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            section = input_grid.extract_subgrid(i*2, j*3, 3, 3)
            density[i][j] = sum(cell == 5 for row in section.values for cell in row)
    return density

def determine_color(density: List[List[int]]) -> int:
    total_sum = sum(sum(row) for row in density)
    non_zero = sum(1 for row in density for cell in row if cell > 0)
    if total_sum <= 10 or non_zero <= 3:
        return 2  # Red
    elif total_sum <= 15 or non_zero <= 6:
        return 1  # Blue
    else:
        return 3  # Green

def create_initial_shape(density: List[List[int]], color: int) -> List[List[int]]:
    threshold = sum(sum(row) for row in density) / 9
    return [[color if cell > threshold else 0 for cell in row] for row in density]

def refine_shape(shape: List[List[int]], density: List[List[int]]) -> List[List[int]]:
    color = max(max(row) for row in shape)
    new_shape = [row[:] for row in shape]
    
    # Preserve corners
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    for i, j in corners:
        if density[i][j] > 0:
            new_shape[i][j] = color
    
    # Preserve edges
    edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
    for i, j in edges:
        if density[i][j] > 1:
            new_shape[i][j] = color
    
    # Handle center
    if density[1][1] > 2:
        new_shape[1][1] = color
    
    return new_shape

def final_adjustments(grid: List[List[int]]) -> List[List[int]]:
    color = max(max(row) for row in grid)
    colored_cells = sum(cell == color for row in grid for cell in row)
    
    if colored_cells == 0:
        grid[1][1] = color
    elif colored_cells == 9:
        grid[1][1] = 0
    elif colored_cells < 3:
        # Add cells to form an L-shape
        grid[0][0] = color
        grid[1][0] = color
        grid[0][1] = color
    elif colored_cells > 7:
        # Remove a cell to create a gap
        if grid[1][1] == color:
            grid[1][1] = 0
        else:
            for i, j in [(0, 1), (1, 0), (1, 2), (2, 1)]:
                if grid[i][j] == color:
                    grid[i][j] = 0
                    break
    
    return grid
