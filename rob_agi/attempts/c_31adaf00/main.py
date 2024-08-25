from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import heapq

def solve_31adaf00(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding blue (1) squares to balance the colors.
    
    The algorithm works as follows:
    1. Creates a deep copy of the input grid and counts gray squares.
    2. Calculates the target number of blue squares.
    3. Identifies key areas for blue square placement using a heatmap.
    4. Grows blue regions organically from these key areas.
    5. Fine-tunes the placement to reach the exact target number of blue squares.
    6. Performs a final balance check and makes minor adjustments if needed.
    
    Returns a new grid with added blue squares while preserving the original gray squares and maintaining visual balance.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    gray_count = count_color(input_grid, 5)
    target_blue = (rows * cols - gray_count) // 2
    
    heatmap = create_heatmap(input_grid)
    blue_count = grow_blue_regions(output_grid, heatmap, target_blue)
    
    if blue_count < target_blue:
        blue_count = fill_remaining_squares(output_grid, blue_count, target_blue)
    elif blue_count > target_blue:
        remove_excess_blue(output_grid, blue_count, target_blue)
    
    final_balance_adjustment(output_grid, target_blue)
    
    return output_grid

def create_heatmap(grid: ColoredGrid) -> List[List[int]]:
    rows, cols = grid.get_dimensions()
    heatmap = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if 0 <= r+dr < rows and 0 <= c+dc < cols:
                            heatmap[r+dr][c+dc] += 1
    
    return heatmap

def grow_blue_regions(grid: ColoredGrid, heatmap: List[List[int]], target: int) -> int:
    rows, cols = grid.get_dimensions()
    blue_count = 0
    visited = set()
    
    while blue_count < target:
        start_r, start_c = max(((r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 0),
                               key=lambda pos: heatmap[pos[0]][pos[1]])
        
        if (start_r, start_c) in visited:
            break
        
        queue = [(start_r, start_c)]
        region_size = 0
        
        while queue and blue_count + region_size < target:
            r, c = queue.pop(0)
            if grid.values[r][c] != 0 or (r, c) in visited:
                continue
            
            grid.values[r][c] = 1
            region_size += 1
            visited.add((r, c))
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 0:
                    queue.append((nr, nc))
        
        blue_count += region_size
    
    return blue_count

def fill_remaining_squares(grid: ColoredGrid, blue_count: int, target: int) -> int:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if blue_count >= target:
                return blue_count
            if grid.values[r][c] == 0:
                grid.values[r][c] = 1
                blue_count += 1
    return blue_count

def final_balance_adjustment(grid: ColoredGrid, target: int) -> None:
    rows, cols = grid.get_dimensions()
    blue_count = count_color(grid, 1)
    
    if blue_count == target:
        return
    
    center_r, center_c = rows // 2, cols // 2
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    cells.sort(key=lambda pos: abs(pos[0] - center_r) + abs(pos[1] - center_c))
    
    for r, c in cells:
        if blue_count < target and grid.values[r][c] == 0:
            grid.values[r][c] = 1
            blue_count += 1
        elif blue_count > target and grid.values[r][c] == 1:
            grid.values[r][c] = 0
            blue_count -= 1
        
        if blue_count == target:
            break

def count_color(grid: ColoredGrid, color: int) -> int:
    return sum(row.count(color) for row in grid.values)

def get_potential_areas(grid: ColoredGrid) -> List[Tuple[float, Tuple[int, int, int, int]]]:
    rows, cols = grid.get_dimensions()
    potential_areas = []
    rectangle_sizes = [(3,3), (2,2)]
    
    for r in range(rows):
        for c in range(cols):
            for width, height in rectangle_sizes:
                if is_valid_blue_area(grid, r, c, width, height):
                    balance_score = calculate_balance_score(grid, r, c, width, height)
                    potential_areas.append((balance_score, (r, c, width, height)))
    
    return sorted(potential_areas, key=lambda x: x[0], reverse=True)

def calculate_balance_score(grid: ColoredGrid, r: int, c: int, width: int, height: int) -> float:
    rows, cols = grid.get_dimensions()
    center_r, center_c = rows // 2, cols // 2
    distance_from_center = ((r + height/2 - center_r)**2 + (c + width/2 - center_c)**2)**0.5
    
    gray_proximity = sum(1 for dr in range(-1, height+1) for dc in range(-1, width+1)
                         if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 5)
    
    return 1 / (1 + distance_from_center) + 0.5 * gray_proximity

def fill_spiral(grid: ColoredGrid, blue_count: int, target_blue: int) -> int:
    rows, cols = grid.get_dimensions()
    center_r, center_c = rows // 2, cols // 2
    spiral = [(0,1), (1,0), (0,-1), (-1,0)]  # right, down, left, up
    r, c = center_r, center_c
    direction = 0
    steps = 0
    max_steps = 1
    
    while blue_count < target_blue:
        if 0 <= r < rows and 0 <= c < cols and grid.values[r][c] == 0:
            grid.values[r][c] = 1
            blue_count += 1
        
        r += spiral[direction][0]
        c += spiral[direction][1]
        steps += 1
        
        if steps == max_steps:
            direction = (direction + 1) % 4
            steps = 0
            if direction % 2 == 0:
                max_steps += 1
    
    return blue_count

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
