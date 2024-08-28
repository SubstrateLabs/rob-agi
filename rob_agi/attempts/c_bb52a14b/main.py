from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional
from collections import Counter

def solve_bb52a14b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the bb52a14b challenge by identifying a unique 3x3 "flower" pattern in the grid and replicating it
    up to two times in suitable areas across the entire grid.

    1. Scan the grid to find the most unique 3x3 pattern based on color distribution and rarity.
    2. Analyze the grid structure and create a color density heat map.
    3. Identify potential replication areas with low color density.
    4. Attempt to replicate the pattern twice, prioritizing areas that improve grid harmony.
    5. If full replication is not possible, attempt partial replication of the most unique elements.
    6. Evaluate the grid after each replication and stop if harmony doesn't improve significantly.
    7. Return the transformed grid with replicated patterns or the original if no replications were possible.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with replicated patterns or the original if no replications were possible.
    """
    pattern, pattern_pos = find_unique_pattern(input_grid)
    if not pattern:
        return input_grid

    output_grid = input_grid.deep_copy()
    heat_map = create_heat_map(output_grid)
    replication_areas = find_replication_areas(output_grid, heat_map, pattern)
    
    original_harmony = calculate_grid_harmony(output_grid)
    
    for _ in range(2):  # Attempt up to two replications
        if not replication_areas:
            break
        
        row, col, _ = replication_areas.pop(0)
        if replicate_pattern(output_grid, pattern, row, col):
            new_harmony = calculate_grid_harmony(output_grid)
            if new_harmony <= original_harmony:
                break  # Stop if harmony doesn't improve
            heat_map = create_heat_map(output_grid)
            replication_areas = find_replication_areas(output_grid, heat_map, pattern)
    
    return output_grid

def find_unique_pattern(grid: ColoredGrid) -> Tuple[Optional[List[List[int]]], Tuple[int, int]]:
    """Find the most unique 3x3 pattern in the grid."""
    rows, cols = grid.get_dimensions()
    best_pattern = None
    best_score = -1
    best_pos = (-1, -1)
    
    for r in range(rows - 2):
        for c in range(cols - 2):
            pattern = extract_pattern(grid, r, c)
            score = calculate_pattern_uniqueness(pattern)
            if score > best_score:
                best_score = score
                best_pattern = pattern
                best_pos = (r, c)
    
    return best_pattern, best_pos

def calculate_pattern_uniqueness(pattern: List[List[int]]) -> float:
    """Calculate the uniqueness score of a pattern based on color distribution and rarity."""
    flat_pattern = [cell for row in pattern for cell in row]
    color_counts = Counter(flat_pattern)
    
    # Assign higher weights to rarer colors
    color_weights = {0: 0, 1: 2, 2: 3, 3: 3, 4: 4, 5: 5, 6: 5, 7: 5, 8: 4, 9: 5}
    
    score = sum(color_weights[color] * (1 / count) for color, count in color_counts.items() if color != 0)
    return score

def create_heat_map(grid: ColoredGrid) -> List[List[float]]:
    """Create a heat map of color density across the grid."""
    rows, cols = grid.get_dimensions()
    heat_map = [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 0:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if 0 <= r + dr < rows and 0 <= c + dc < cols:
                            heat_map[r + dr][c + dc] += 1
    
    return heat_map

def find_replication_areas(grid: ColoredGrid, heat_map: List[List[float]], pattern: List[List[int]]) -> List[Tuple[int, int, float]]:
    """Find potential replication areas based on heat map and pattern compatibility."""
    rows, cols = grid.get_dimensions()
    areas = []
    for r in range(rows - 2):
        for c in range(cols - 2):
            score = calculate_replication_score(grid, pattern, r, c, heat_map)
            if score > 0:
                areas.append((r, c, score))
    return sorted(areas, key=lambda x: -x[2])  # Sort by score in descending order

def calculate_replication_score(grid: ColoredGrid, pattern: List[List[int]], start_r: int, start_c: int, heat_map: List[List[float]]) -> float:
    """Calculate the replication score for a potential area."""
    score = 0
    heat = sum(heat_map[start_r + r][start_c + c] for r in range(3) for c in range(3))
    
    for r in range(3):
        for c in range(3):
            grid_value = grid.get_cell(start_r + r, start_c + c)
            pattern_value = pattern[r][c]
            if grid_value == 0 and pattern_value != 0:
                score += 2  # Prefer filling black spaces
            elif grid_value != 0 and pattern_value != 0:
                if grid_value == pattern_value:
                    score += 1  # Matching colors
                else:
                    return -1  # Area is not suitable for replication
    
    return score * (1 / (heat + 1))  # Adjust score based on heat (lower heat is better)

def replicate_pattern(grid: ColoredGrid, pattern: List[List[int]], start_r: int, start_c: int) -> bool:
    """Replicate the given pattern at the specified position in the grid, allowing partial replication."""
    full_replication = True
    for r in range(3):
        for c in range(3):
            grid_value = grid.get_cell(start_r + r, start_c + c)
            pattern_value = pattern[r][c]
            if grid_value == 0 or grid_value == pattern_value:
                grid.set_cell(start_r + r, start_c + c, pattern_value)
            else:
                full_replication = False
    return full_replication

def calculate_grid_harmony(grid: ColoredGrid) -> float:
    """Calculate the harmony score of the grid based on color distribution and pattern repetition."""
    rows, cols = grid.get_dimensions()
    color_counts = Counter(grid.get_cell(r, c) for r in range(rows) for c in range(cols) if grid.get_cell(r, c) != 0)
    
    color_variety = len(color_counts)
    color_balance = sum(count * count for count in color_counts.values())
    pattern_repetition = sum(1 for r in range(rows - 2) for c in range(cols - 2) 
                             if is_flower_pattern(extract_pattern(grid, r, c)))
    
    return color_variety * pattern_repetition / (color_balance + 1)

def extract_pattern(grid: ColoredGrid, start_r: int, start_c: int) -> List[List[int]]:
    """Extract a 3x3 pattern starting from the given position."""
    return [[grid.get_cell(r, c) for c in range(start_c, start_c + 3)] for r in range(start_r, start_r + 3)]

def is_flower_pattern(pattern: List[List[int]]) -> bool:
    """Check if the given pattern is a flower pattern."""
    center = pattern[1][1]
    if center == 0:
        return False
    surrounding = [pattern[i][j] for i in range(3) for j in range(3) if (i, j) != (1, 1)]
    return len(set(surrounding)) <= 2 and all(color != 0 for color in surrounding)
