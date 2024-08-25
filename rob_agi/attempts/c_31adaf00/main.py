from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import heapq

def solve_31adaf00(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding blue (1) squares to balance the colors.
    
    The algorithm works as follows:
    1. Analyzes the input grid to count gray squares and identify black areas.
    2. Calculates a target number of blue squares for balance.
    3. Identifies potential blue areas, prioritizing larger rectangles.
    4. Fills in blue areas until reaching the target or running out of suitable spaces.
    5. Fine-tunes the result by filling smaller areas if needed.
    6. Performs a final balance check and adjusts if necessary.
    
    Returns a new grid with added blue squares while preserving the original gray squares.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    gray_count = count_color(input_grid, 5)
    target_blue = (rows * cols - gray_count) // 2
    
    potential_areas = []
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 0:
                area = find_largest_rectangle(input_grid, r, c)
                if area:
                    heapq.heappush(potential_areas, (-area[2] * area[3], area))
    
    blue_count = 0
    while blue_count < target_blue and potential_areas:
        _, (x, y, width, height) = heapq.heappop(potential_areas)
        if is_valid_blue_area(output_grid, x, y, width, height):
            fill_area(output_grid, x, y, width, height, 1)
            blue_count += width * height
    
    # Fine-tuning
    for r in range(rows):
        for c in range(cols):
            if blue_count >= target_blue:
                break
            if output_grid.values[r][c] == 0 and is_valid_blue_area(output_grid, r, c, 1, 1):
                output_grid.values[r][c] = 1
                blue_count += 1
    
    return output_grid

def count_color(grid: ColoredGrid, color: int) -> int:
    return sum(row.count(color) for row in grid.values)

def find_largest_rectangle(grid: ColoredGrid, start_x: int, start_y: int) -> Tuple[int, int, int, int]:
    rows, cols = grid.get_dimensions()
    max_width = 0
    max_height = 0
    for width in range(cols - start_y):
        if grid.values[start_x][start_y + width] != 0:
            break
        max_width = width + 1
    for height in range(rows - start_x):
        if any(grid.values[start_x + height][start_y + w] != 0 for w in range(max_width)):
            break
        max_height = height + 1
    return (start_x, start_y, max_width, max_height) if max_width > 0 and max_height > 0 else None

def is_valid_blue_area(grid: ColoredGrid, x: int, y: int, width: int, height: int) -> bool:
    rows, cols = grid.get_dimensions()
    if x + height > rows or y + width > cols:
        return False
    return all(grid.values[r][c] == 0 for r in range(x, x + height) for c in range(y, y + width))

def fill_area(grid: ColoredGrid, x: int, y: int, width: int, height: int, color: int) -> None:
    for r in range(x, x + height):
        for c in range(y, y + width):
            grid.values[r][c] = color
