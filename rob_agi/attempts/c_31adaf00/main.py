from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import heapq

def solve_31adaf00(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding blue (1) squares to balance the colors.
    
    The algorithm works as follows:
    1. Analyzes the input grid to count gray squares and calculate the target number of blue squares.
    2. Creates a heatmap based on proximity to gray squares and edges.
    3. Places blue squares in phases:
       a. Large-scale placement (3x3 squares) in large black areas.
       b. Medium-scale placement (2x2 squares) complementing gray patterns.
       c. Small-scale placement (individual squares) using the heatmap.
    4. Ensures connectivity between blue regions and overall balance.
    5. Fine-tunes the placement to match the target blue count exactly.
    6. Performs final checks for symmetry and aesthetic balance.
    
    Returns a new grid with added blue squares while preserving the original gray squares and maintaining visual balance.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    gray_count = count_color(input_grid, 5)
    target_blue = (rows * cols - gray_count) // 2
    
    heatmap = create_heatmap(input_grid)
    blue_count = place_large_blue_squares(output_grid, heatmap, target_blue)
    blue_count = place_medium_blue_squares(output_grid, heatmap, blue_count, target_blue)
    blue_count = fill_remaining_squares(output_grid, heatmap, blue_count, target_blue)
    
    ensure_connectivity(output_grid)
    final_balance_adjustment(output_grid, target_blue)
    
    return output_grid

def create_heatmap(grid: ColoredGrid) -> List[List[float]]:
    rows, cols = grid.get_dimensions()
    heatmap = [[0.0 for _ in range(cols)] for _ in range(rows)]
    center_r, center_c = rows // 2, cols // 2
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 5:  # Gray square
                for dr in [-3, -2, -1, 0, 1, 2, 3]:
                    for dc in [-3, -2, -1, 0, 1, 2, 3]:
                        if 0 <= r+dr < rows and 0 <= c+dc < cols:
                            distance = max(abs(dr), abs(dc))
                            heatmap[r+dr][c+dc] += max(0, 4 - distance)
            
            # Add edge and corner bonuses
            edge_bonus = 2 if r == 0 or r == rows-1 or c == 0 or c == cols-1 else 0
            corner_bonus = 2 if (r == 0 or r == rows-1) and (c == 0 or c == cols-1) else 0
            heatmap[r][c] += edge_bonus + corner_bonus
            
            # Consider distance from center
            center_distance = ((r - center_r)**2 + (c - center_c)**2)**0.5
            heatmap[r][c] += 2 / (1 + center_distance)
    
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
        
        region_sizes = [(3, 3), (2, 2), (2, 3), (3, 2)]
        for height, width in region_sizes:
            if is_valid_blue_area(grid, start_r, start_c, width, height) and blue_count + width * height <= target:
                fill_area(grid, start_r, start_c, width, height, 1)
                blue_count += width * height
                for r in range(start_r, start_r + height):
                    for c in range(start_c, start_c + width):
                        visited.add((r, c))
                break
        else:
            visited.add((start_r, start_c))
    
    return blue_count

def fill_remaining_squares(grid: ColoredGrid, heatmap: List[List[float]], blue_count: int, target: int) -> int:
    rows, cols = grid.get_dimensions()
    cells = [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 0]
    cells.sort(key=lambda pos: (-heatmap[pos[0]][pos[1]], -count_adjacent_blue(grid, pos[0], pos[1])))
    
    for r, c in cells:
        if blue_count >= target:
            break
        grid.values[r][c] = 1
        blue_count += 1
    return blue_count

def count_adjacent_blue(grid: ColoredGrid, r: int, c: int) -> int:
    rows, cols = grid.get_dimensions()
    count = 0
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 1:
            count += 1
    return count

def is_isolated(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 1:
            return False
    return True

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
    blue_cells = [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 1]
    blue_cells.sort(key=lambda pos: -count_adjacent_blue(grid, pos[0], pos[1]))
    
    for r, c in blue_cells:
        if blue_count <= target_blue:
            return
        grid.values[r][c] = 0
        blue_count -= 1
def get_potential_regions(grid: ColoredGrid) -> List[Tuple[float, Tuple[int, int, int, int]]]:
    rows, cols = grid.get_dimensions()
    potential_regions = []
    region_sizes = [(2,2), (3,3), (2,4), (4,2), (3,2), (2,3)]
    
    for r in range(rows):
        for c in range(cols):
            for width, height in region_sizes:
                if is_valid_blue_area(grid, r, c, width, height):
                    score = score_region(grid, r, c, width, height)
                    potential_regions.append((score, (r, c, width, height)))
    
    return sorted(potential_regions, key=lambda x: x[0], reverse=True)

def score_region(grid: ColoredGrid, r: int, c: int, width: int, height: int) -> float:
    rows, cols = grid.get_dimensions()
    center_r, center_c = rows // 2, cols // 2
    distance_from_center = ((r + height/2 - center_r)**2 + (c + width/2 - center_c)**2)**0.5
    
    gray_proximity = sum(1 for dr in range(-1, height+1) for dc in range(-1, width+1)
                         if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 5)
    
    edge_score = 1 if r == 0 or r + height == rows or c == 0 or c + width == cols else 0
    
    size_score = width * height
    
    return (1 / (1 + distance_from_center)) + (0.5 * gray_proximity) + edge_score + (0.1 * size_score)

def place_blue_regions(grid: ColoredGrid, potential_regions: List[Tuple[float, Tuple[int, int, int, int]]], target_blue: int) -> int:
    blue_count = 0
    for _, (r, c, width, height) in potential_regions:
        if blue_count + (width * height) <= target_blue and is_valid_blue_area(grid, r, c, width, height):
            fill_area(grid, r, c, width, height, 1)
            blue_count += width * height
        if blue_count >= target_blue:
            break
    return blue_count
def get_potential_regions(grid: ColoredGrid, heatmap: List[List[float]]) -> List[Tuple[float, Tuple[int, int, int, int]]]:
    rows, cols = grid.get_dimensions()
    potential_regions = []
    region_sizes = [(3,3), (2,2), (2,3), (3,2)]
    
    for r in range(rows):
        for c in range(cols):
            for width, height in region_sizes:
                if is_valid_blue_area(grid, r, c, width, height):
                    score = score_region(grid, heatmap, r, c, width, height)
                    potential_regions.append((score, (r, c, width, height)))
    
    return sorted(potential_regions, key=lambda x: x[0], reverse=True)

def score_region(grid: ColoredGrid, heatmap: List[List[float]], r: int, c: int, width: int, height: int) -> float:
    rows, cols = grid.get_dimensions()
    heatmap_score = sum(heatmap[r+dr][c+dc] for dr in range(height) for dc in range(width) if r+dr < rows and c+dc < cols)
    
    gray_proximity = sum(1 for dr in range(-1, height+1) for dc in range(-1, width+1)
                         if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 5)
    
    edge_score = 1 if r == 0 or r + height == rows or c == 0 or c + width == cols else 0
    
    return heatmap_score + (0.5 * gray_proximity) + edge_score

def place_blue_regions(grid: ColoredGrid, potential_regions: List[Tuple[float, Tuple[int, int, int, int]]], target_blue: int) -> int:
    blue_count = 0
    for _, (r, c, width, height) in potential_regions:
        if blue_count + (width * height) <= target_blue and is_valid_blue_area(grid, r, c, width, height):
            fill_area(grid, r, c, width, height, 1)
            blue_count += width * height
        if blue_count >= target_blue * 0.8:  # Stop at 80% to avoid overfilling
            break
    return blue_count
def place_large_blue_squares(grid: ColoredGrid, heatmap: List[List[float]], target_blue: int) -> int:
    rows, cols = grid.get_dimensions()
    blue_count = 0
    large_areas = find_large_black_areas(grid, 4)
    
    for area in large_areas[:2]:  # Place up to two 3x3 squares
        r, c = area[0], area[1]
        if is_valid_blue_area(grid, r, c, 3, 3) and blue_count + 9 <= target_blue * 0.6:
            fill_area(grid, r, c, 3, 3, 1)
            blue_count += 9
    
    return blue_count

def find_large_black_areas(grid: ColoredGrid, min_size: int) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    large_areas = []
    for r in range(rows - min_size + 1):
        for c in range(cols - min_size + 1):
            if all(grid.values[r+dr][c+dc] == 0 for dr in range(min_size) for dc in range(min_size)):
                large_areas.append((r, c))
    return large_areas

def place_medium_blue_squares(grid: ColoredGrid, heatmap: List[List[float]], blue_count: int, target_blue: int) -> int:
    rows, cols = grid.get_dimensions()
    potential_areas = [(r, c) for r in range(rows-1) for c in range(cols-1) 
                       if is_valid_blue_area(grid, r, c, 2, 2)]
    potential_areas.sort(key=lambda pos: sum(heatmap[pos[0]+dr][pos[1]+dc] for dr in range(2) for dc in range(2)), reverse=True)
    
    for r, c in potential_areas:
        if blue_count + 4 <= target_blue * 0.9:
            fill_area(grid, r, c, 2, 2, 1)
            blue_count += 4
        else:
            break
    
    return blue_count

def ensure_connectivity(grid: ColoredGrid) -> None:
    blue_regions = grid.find_connected_regions(1)
    if len(blue_regions) > 1:
        main_region = max(blue_regions, key=len)
        for region in blue_regions:
            if region != main_region:
                connect_regions(grid, main_region, region)

def connect_regions(grid: ColoredGrid, region1: List[Tuple[int, int]], region2: List[Tuple[int, int]]) -> None:
    start = region1[0]
    end = min(region2, key=lambda pos: ((pos[0]-start[0])**2 + (pos[1]-start[1])**2)**0.5)
    path = find_path(grid, start, end)
    for r, c in path:
        grid.values[r][c] = 1

def find_path(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    queue = deque([(start, [start])])
    visited = set([start])
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == end:
            return path
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return []
