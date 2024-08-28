from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import random

def solve_7ee1c6ea(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by redistributing colors within connected regions while preserving key structures.
    
    1. Identifies and preserves key structures (frames, crosses, border-like patterns).
    2. Segments the grid into regions based on these structures.
    3. For each region, redistributes colors to maintain overall ratios and create larger contiguous areas.
    4. Preserves black (0) and gray (5) squares.
    5. Avoids creating 2x2 squares of the same color.
    6. Refines small isolated color areas.
    7. Balances changes across regions to maintain overall grid coherence.
    8. Iterates the process until stability or max iterations.
    
    Returns the transformed grid with improved color distribution and pattern coherence.
    """
    new_grid = input_grid.deep_copy()
    key_structures = identify_key_structures(new_grid)
    regions = segment_grid(new_grid, key_structures)
    
    initial_color_ratios = calculate_color_ratios(new_grid)
    
    max_iterations = 10
    for _ in range(max_iterations):
        changed = False
        for region in regions:
            if redistribute_colors_in_region(new_grid, region):
                changed = True
        if not changed:
            break
    
    refine_small_areas(new_grid)
    balance_changes(new_grid, initial_color_ratios)
    return new_grid

def calculate_color_ratios(grid: ColoredGrid) -> Dict[int, float]:
    color_counts = defaultdict(int)
    total_cells = 0
    for row in grid.values:
        for cell in row:
            if cell not in [0, 5]:  # Exclude black and gray
                color_counts[cell] += 1
                total_cells += 1
    return {color: count / total_cells for color, count in color_counts.items()}

def redistribute_colors_in_region(grid: ColoredGrid, region: Set[Tuple[int, int]]) -> bool:
    color_areas = defaultdict(set)
    for r, c in region:
        color = grid.values[r][c]
        if color not in [0, 5]:  # Exclude black and gray
            color_areas[color].add((r, c))
    
    if len(color_areas) < 2:
        return False
    
    changed = False
    for color, area in sorted(color_areas.items(), key=lambda x: len(x[1]), reverse=True):
        adjacent_colors = find_adjacent_colors(grid, area)
        for adj_color in adjacent_colors:
            if len(color_areas[adj_color]) < len(area):
                expansion = expand_color_area(grid, area, adj_color)
                if expansion:
                    changed = True
                    color_areas[color] -= expansion
                    color_areas[adj_color] |= expansion
                    for r, c in expansion:
                        grid.values[r][c] = adj_color
    
    return changed

def expand_color_area(grid: ColoredGrid, area: Set[Tuple[int, int]], target_color: int) -> Set[Tuple[int, int]]:
    expansion = set()
    rows, cols = grid.get_dimensions()
    for r, c in area:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == target_color:
                if is_valid_change(grid, (nr, nc), grid.values[r][c]):
                    expansion.add((nr, nc))
    return expansion

def balance_changes(grid: ColoredGrid, initial_ratios: Dict[int, float]):
    current_ratios = calculate_color_ratios(grid)
    rows, cols = grid.get_dimensions()
    
    for color, initial_ratio in initial_ratios.items():
        current_ratio = current_ratios.get(color, 0)
        if abs(current_ratio - initial_ratio) > 0.05:  # 5% threshold
            target_count = int(rows * cols * initial_ratio)
            current_count = int(rows * cols * current_ratio)
            
            if current_count < target_count:
                # Need to increase this color
                for _ in range(target_count - current_count):
                    for r in range(rows):
                        for c in range(cols):
                            if grid.values[r][c] not in [0, 5] and is_valid_change(grid, (r, c), color):
                                grid.values[r][c] = color
                                break
                        else:
                            continue
                        break
            else:
                # Need to decrease this color
                for _ in range(current_count - target_count):
                    for r in range(rows):
                        for c in range(cols):
                            if grid.values[r][c] == color:
                                for new_color in initial_ratios:
                                    if new_color != color and is_valid_change(grid, (r, c), new_color):
                                        grid.values[r][c] = new_color
                                        break
                                break
                        else:
                            continue
                        break

def identify_key_structures(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    structures = []
    rows, cols = grid.get_dimensions()
    
    # Identify frame
    frame = set()
    for r in range(rows):
        for c in range(cols):
            if r in (0, rows-1) or c in (0, cols-1):
                if grid.values[r][c] == 5:  # Gray
                    frame.add((r, c))
    if frame:
        structures.append(frame)
    
    # Identify cross
    horizontal = set((r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 5)
    vertical = set((r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 5)
    cross = horizontal.intersection(vertical)
    if len(cross) > 0:
        structures.append(cross)
    
    return structures

def segment_grid(grid: ColoredGrid, key_structures: List[Set[Tuple[int, int]]]) -> List[Set[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    all_cells = set((r, c) for r in range(rows) for c in range(cols))
    structure_cells = set.union(*key_structures) if key_structures else set()
    remaining_cells = all_cells - structure_cells
    
    regions = []
    while remaining_cells:
        start = remaining_cells.pop()
        region = find_connected_region(grid, start[0], start[1], grid.values[start[0]][start[1]])
        regions.append(region)
        remaining_cells -= region
    
    return regions + key_structures

def find_connected_region(grid: ColoredGrid, r: int, c: int, color: int) -> Set[Tuple[int, int]]:
    region = set()
    stack = [(r, c)]
    rows, cols = grid.get_dimensions()
    while stack:
        curr_r, curr_c = stack.pop()
        if (curr_r, curr_c) in region:
            continue
        if 0 <= curr_r < rows and 0 <= curr_c < cols and grid.values[curr_r][curr_c] == color:
            region.add((curr_r, curr_c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((curr_r + dr, curr_c + dc))
    return region

def redistribute_colors_in_region(grid: ColoredGrid, region: Set[Tuple[int, int]]) -> bool:
    color_areas = defaultdict(set)
    for r, c in region:
        color = grid.values[r][c]
        if color not in [0, 5]:  # Exclude black and gray
            color_areas[color].add((r, c))
    
    if len(color_areas) < 2:
        return False
    
    changed = False
    for color, area in sorted(color_areas.items(), key=lambda x: len(x[1]), reverse=True):
        adjacent_colors = find_adjacent_colors(grid, area)
        for adj_color in adjacent_colors:
            if len(color_areas[adj_color]) < len(area):
                expansion = expand_color_area(grid, area, adj_color)
                if expansion:
                    changed = True
                    color_areas[color] -= expansion
                    color_areas[adj_color] |= expansion
                    for r, c in expansion:
                        grid.values[r][c] = adj_color
    
    return changed

def find_adjacent_colors(grid: ColoredGrid, area: Set[Tuple[int, int]]) -> Set[int]:
    adjacent_colors = set()
    rows, cols = grid.get_dimensions()
    for r, c in area:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in area:
                adj_color = grid.values[nr][nc]
                if adj_color not in [0, 5]:  # Exclude black and gray
                    adjacent_colors.add(adj_color)
    return adjacent_colors

def expand_color_area(grid: ColoredGrid, area: Set[Tuple[int, int]], target_color: int) -> Set[Tuple[int, int]]:
    expansion = set()
    rows, cols = grid.get_dimensions()
    for r, c in area:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == target_color:
                if is_valid_change(grid, (nr, nc), grid.values[r][c]):
                    expansion.add((nr, nc))
    return expansion

def is_valid_change(grid: ColoredGrid, sq: Tuple[int, int], new_color: int) -> bool:
    r, c = sq
    rows, cols = grid.get_dimensions()
    if grid.values[r][c] in [0, 5]:
        return False
    
    for dr, dc in [(0, 1), (1, 0), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if all(grid.values[r+i][c+j] == new_color for i in range(2) for j in range(2) if 0 <= r+i < rows and 0 <= c+j < cols):
                return False
    
    return True

def refine_small_areas(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] not in [0, 5]:
                area = find_connected_region(grid, r, c, grid.values[r][c])
                if 1 < len(area) <= 3:
                    merge_small_area(grid, area)

def merge_small_area(grid: ColoredGrid, area: Set[Tuple[int, int]]):
    adjacent_colors = find_adjacent_colors(grid, area)
    if adjacent_colors:
        new_color = random.choice(list(adjacent_colors))
        for r, c in area:
            if is_valid_change(grid, (r, c), new_color):
                grid.values[r][c] = new_color
