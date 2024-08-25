from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import heapq

def solve_31adaf00(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding blue (1) squares to balance the colors.
    
    The algorithm works as follows:
    1. Creates a deep copy of the input grid and counts gray squares.
    2. Calculates the target number of blue squares.
    3. Identifies potential locations for blue rectangles, prioritizing corners and edges.
    4. Fills blue rectangles up to 3x3 in size until reaching the target or running out of suitable spaces.
    5. Distributes any remaining blue squares across the grid.
    6. Performs a final adjustment if needed to match the exact target.
    
    Returns a new grid with added blue squares while preserving the original gray squares.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    gray_count = count_color(input_grid, 5)
    target_blue = (rows * cols - gray_count) // 2
    
    potential_areas = get_potential_areas(input_grid)
    
    blue_count = 0
    while blue_count < target_blue and potential_areas:
        _, (x, y, width, height) = heapq.heappop(potential_areas)
        if is_valid_blue_area(output_grid, x, y, width, height):
            fill_area(output_grid, x, y, width, height, 1)
            blue_count += width * height
    
    # Distribute remaining blue squares
    if blue_count < target_blue:
        blue_count = distribute_remaining_blue(output_grid, blue_count, target_blue)
    
    # Final adjustment
    if blue_count > target_blue:
        remove_excess_blue(output_grid, blue_count, target_blue)
    
    return output_grid

def count_color(grid: ColoredGrid, color: int) -> int:
    return sum(row.count(color) for row in grid.values)

def get_potential_areas(grid: ColoredGrid) -> List[Tuple[int, Tuple[int, int, int, int]]]:
    rows, cols = grid.get_dimensions()
    potential_areas = []
    rectangle_sizes = [(3,3), (3,2), (2,3), (2,2), (3,1), (1,3), (2,1), (1,2), (1,1)]
    corners = [(0,0), (0,cols-1), (rows-1,0), (rows-1,cols-1)]
    edges = [(0, range(1,cols-1)), (rows-1, range(1,cols-1)), (range(1,rows-1), 0), (range(1,rows-1), cols-1)]
    
    # Check corners
    for x, y in corners:
        for width, height in rectangle_sizes:
            if is_valid_blue_area(grid, x, y, width, height):
                priority = -(width * height * 1000 + x * 10 + y)
                heapq.heappush(potential_areas, (priority, (x, y, width, height)))
    
    # Check edges
    for r, c_range in edges[:2]:
        for c in c_range:
            for width, height in rectangle_sizes:
                if is_valid_blue_area(grid, r, c, width, height):
                    priority = -(width * height * 100 + r * 10 + c)
                    heapq.heappush(potential_areas, (priority, (r, c, width, height)))
    for r_range, c in edges[2:]:
        for r in r_range:
            for width, height in rectangle_sizes:
                if is_valid_blue_area(grid, r, c, width, height):
                    priority = -(width * height * 100 + r * 10 + c)
                    heapq.heappush(potential_areas, (priority, (r, c, width, height)))
    
    # Check interior
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            for width, height in rectangle_sizes:
                if is_valid_blue_area(grid, r, c, width, height):
                    priority = -(width * height * 10 + r * 10 + c)
                    heapq.heappush(potential_areas, (priority, (r, c, width, height)))
    
    return potential_areas

def is_valid_blue_area(grid: ColoredGrid, x: int, y: int, width: int, height: int) -> bool:
    rows, cols = grid.get_dimensions()
    if x + height > rows or y + width > cols:
        return False
    return all(grid.values[r][c] == 0 for r in range(x, x + height) for c in range(y, y + width))

def fill_area(grid: ColoredGrid, x: int, y: int, width: int, height: int, color: int) -> None:
    for r in range(x, x + height):
        for c in range(y, y + width):
            grid.values[r][c] = color

def distribute_remaining_blue(grid: ColoredGrid, blue_count: int, target_blue: int) -> int:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols) if r % 2 == 0 else range(cols-1, -1, -1):
            if blue_count >= target_blue:
                return blue_count
            if grid.values[r][c] == 0:
                grid.values[r][c] = 1
                blue_count += 1
    return blue_count

def remove_excess_blue(grid: ColoredGrid, blue_count: int, target_blue: int) -> None:
    rows, cols = grid.get_dimensions()
    for r in range(rows-1, -1, -1):
        for c in range(cols-1, -1, -1) if r % 2 == 0 else range(cols):
            if blue_count <= target_blue:
                return
            if grid.values[r][c] == 1:
                grid.values[r][c] = 0
                blue_count -= 1
