from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ed74f2f2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 9x5 input grid into a 3x3 output grid based on the following rules:
    1. Creates a density grid by analyzing 3x3 sections of the input grid.
    2. Determines the output color based on the density and distribution of gray cells.
    3. Creates an initial shape based on the density grid and chosen color.
    4. Applies smoothing rules to ensure connectivity and remove isolated cells.
    5. Makes final adjustments to ensure a valid and interesting output.
    6. Returns the final 3x3 ColoredGrid output.
    """
    density_grid = create_density_grid(input_grid)
    color = determine_color(density_grid)
    initial_shape = create_initial_shape(density_grid, color)
    smoothed_shape = smooth_grid(initial_shape)
    final_shape = final_adjustments(smoothed_shape)
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
    if total_sum <= 12 or non_zero <= 4:
        return 2  # Red
    elif total_sum <= 18 or non_zero <= 7:
        return 1  # Blue
    else:
        return 3  # Green

def create_initial_shape(density: List[List[int]], color: int) -> List[List[int]]:
    threshold = sum(sum(row) for row in density) / 9
    return [[color if cell > threshold else 0 for cell in row] for row in density]

def smooth_grid(grid: List[List[int]]) -> List[List[int]]:
    new_grid = [row[:] for row in grid]
    color = max(max(row) for row in grid)  # Get the non-zero color
    
    # First pass: remove isolated cells
    for i in range(3):
        for j in range(3):
            if new_grid[i][j] == color and count_adjacent_color(new_grid, i, j, color) < 2:
                new_grid[i][j] = 0
    
    # Second pass: fill gaps
    for i in range(3):
        for j in range(3):
            if new_grid[i][j] == 0 and count_adjacent_color(new_grid, i, j, color) >= 2:
                new_grid[i][j] = color
    
    return new_grid

def count_adjacent_color(grid: List[List[int]], i: int, j: int, color: int) -> int:
    count = 0
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < 3 and 0 <= nj < 3 and grid[ni][nj] == color:
            count += 1
    return count

def final_adjustments(grid: List[List[int]]) -> List[List[int]]:
    color = max(max(row) for row in grid)  # Get the non-zero color
    colored_cells = sum(cell == color for row in grid for cell in row)
    
    if colored_cells == 0:
        grid[1][1] = color
    elif colored_cells == 9:
        grid[1][1] = 0
    elif colored_cells < 4:
        # Add a cell to make the shape more interesting
        for i, j in [(0, 1), (1, 0), (1, 2), (2, 1)]:
            if grid[i][j] == 0:
                grid[i][j] = color
                break
    elif colored_cells > 6:
        # Remove a cell to make the shape more interesting
        for i, j in [(0, 1), (1, 0), (1, 2), (2, 1)]:
            if grid[i][j] == color:
                grid[i][j] = 0
                break
    
    return grid
