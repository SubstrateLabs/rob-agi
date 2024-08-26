from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ed74f2f2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 9x5 input grid into a 3x3 output grid based on the following rules:
    1. Creates a density grid by analyzing 3x3 sections of the input grid.
    2. Determines the output color based on the density and distribution of gray cells.
    3. Creates an initial shape based on the density grid and chosen color.
    4. Applies smoothing rules to remove isolated cells and fill gaps.
    5. Makes final adjustments to ensure a valid and interesting output.
    6. Returns the final 3x3 ColoredGrid output.
    """
    density_grid = create_density_grid(input_grid)
    color = determine_color(density_grid)
    initial_shape = create_initial_shape(density_grid, color)
    smoothed_shape = smooth_grid(initial_shape, color)
    final_shape = final_adjustments(smoothed_shape, color)
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
    if total_sum < 10 or non_zero <= 3:
        return 2  # Red
    elif total_sum < 15 and non_zero <= 6:
        return 3  # Green
    else:
        return 1  # Blue

def create_initial_shape(density: List[List[int]], color: int) -> List[List[int]]:
    threshold = sum(sum(row) for row in density) / 9
    return [[color if cell >= threshold else 0 for cell in row] for row in density]

def smooth_grid(grid: List[List[int]], color: int) -> List[List[int]]:
    new_grid = [row[:] for row in grid]
    for i in range(3):
        for j in range(3):
            if new_grid[i][j] == color and not has_adjacent_color(new_grid, i, j, color):
                new_grid[i][j] = 0
            elif new_grid[i][j] == 0 and is_surrounded_by_color(new_grid, i, j, color):
                new_grid[i][j] = color
    return new_grid

def has_adjacent_color(grid: List[List[int]], i: int, j: int, color: int) -> bool:
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < 3 and 0 <= nj < 3 and grid[ni][nj] == color:
            return True
    return False

def is_surrounded_by_color(grid: List[List[int]], i: int, j: int, color: int) -> bool:
    return all(grid[i+di][j+dj] == color for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)] if 0 <= i+di < 3 and 0 <= j+dj < 3)

def final_adjustments(grid: List[List[int]], color: int) -> List[List[int]]:
    if all(cell == 0 for row in grid for cell in row):
        grid[1][1] = color
    elif all(cell == color for row in grid for cell in row):
        grid[1][1] = 0
    return grid
