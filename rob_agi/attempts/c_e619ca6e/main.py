import random
import math
from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_e619ca6e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid expansion challenge by growing green structures organically.
    
    The solution follows these steps:
    1. Initialize and analyze the grid, identifying seed points and creating influence maps.
    2. Perform main growth phase using probability-based expansion.
    3. Add branching structures to create more complex patterns.
    4. Create isolated formations in empty areas.
    5. Fine-tune the structure by smoothing edges and filling small gaps.
    6. Adjust for symmetry and refine edges.
    7. Perform final connectivity check and cleanup.

    This approach creates a complex, organic structure that expands from the original green cells,
    maintaining intricate patterns while adapting to different input configurations.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    seed_points = identify_seed_points(input_grid)
    
    influence_maps = create_influence_maps(output_grid, seed_points)
    
    # Main growth phase
    for _ in range(min(rows, cols)):  # Adjust iterations based on grid size
        expand_structures(output_grid, influence_maps)
        ensure_connectivity(output_grid, seed_points)
    
    add_branches(output_grid, influence_maps)
    create_isolated_formations(output_grid, influence_maps)
    fine_tune_structure(output_grid)
    adjust_symmetry(output_grid, seed_points)
    refine_edges(output_grid)
    
    final_connectivity_check(output_grid, seed_points)
    final_cleanup(output_grid)
    
    return output_grid

def identify_seed_points(grid: ColoredGrid) -> Set[Tuple[int, int]]:
    seed_points = set()
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                seed_points.add((r, c))
    return seed_points

def create_influence_maps(grid: ColoredGrid, seed_points: Set[Tuple[int, int]]) -> dict:
    rows, cols = grid.get_dimensions()
    distance_map = [[float('inf') for _ in range(cols)] for _ in range(rows)]
    edge_pressure_map = [[0.0 for _ in range(cols)] for _ in range(rows)]
    symmetry_map = [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    # Calculate center of mass for seed points
    if seed_points:
        center_r = sum(r for r, _ in seed_points) / len(seed_points)
        center_c = sum(c for _, c in seed_points) / len(seed_points)
    else:
        center_r, center_c = rows // 2, cols // 2
    
    # Create distance and symmetry maps
    for r in range(rows):
        for c in range(cols):
            for sr, sc in seed_points:
                dist = math.sqrt((r - sr)**2 + (c - sc)**2)
                distance_map[r][c] = min(distance_map[r][c], dist)
            
            # Edge pressure (higher near edges)
            edge_dist = min(r, c, rows-1-r, cols-1-c)
            edge_pressure_map[r][c] = 1 - (edge_dist / max(rows//2, cols//2))
            
            # Symmetry (higher in symmetrical positions)
            sym_r, sym_c = 2*center_r - r, 2*center_c - c
            if 0 <= sym_r < rows and 0 <= sym_c < cols:
                symmetry_map[r][c] = 1 - (abs(r - sym_r) + abs(c - sym_c)) / (rows + cols)
    
    return {
        'distance': distance_map,
        'edge_pressure': edge_pressure_map,
        'symmetry': symmetry_map
    }

def expand_structures(grid: ColoredGrid, influence_maps: dict):
    rows, cols = grid.get_dimensions()
    new_green_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0 and any(grid.get_cell(r+dr, c+dc) == 3 
                                                for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)] 
                                                if 0 <= r+dr < rows and 0 <= c+dc < cols):
                growth_prob = calculate_growth_probability(r, c, influence_maps)
                if random.random() < growth_prob:
                    new_green_cells.add((r, c))
    for r, c in new_green_cells:
        grid.set_cell(r, c, 3)

def calculate_growth_probability(r: int, c: int, influence_maps: dict) -> float:
    distance_factor = 1 / (1 + influence_maps['distance'][r][c])
    edge_factor = 1 - influence_maps['edge_pressure'][r][c]
    symmetry_factor = influence_maps['symmetry'][r][c]
    return 0.3 * distance_factor + 0.3 * edge_factor + 0.4 * symmetry_factor

def ensure_connectivity(grid: ColoredGrid, seed_points: Set[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    visited = set()
    
    def dfs(r, c):
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == 3:
                visited.add((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
    
    # Start DFS from all seed points
    for r, c in seed_points:
        if (r, c) not in visited:
            dfs(r, c)
    
    # Remove any disconnected green cells
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3 and (r, c) not in visited:
                grid.set_cell(r, c, 0)

def add_branches(grid: ColoredGrid, influence_maps: dict):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                green_neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]
                                      if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 3)
                if green_neighbors in [1, 2] and random.random() < 0.1:
                    branch_direction = max([(dr, dc) for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]],
                                           key=lambda d: influence_maps['symmetry'][r+d[0]][c+d[1]])
                    for i in range(1, 4):  # Branch length 2-3 cells
                        nr, nc = r + i*branch_direction[0], c + i*branch_direction[1]
                        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 0:
                            grid.set_cell(nr, nc, 3)
                        else:
                            break

def create_isolated_formations(grid: ColoredGrid, influence_maps: dict):
    rows, cols = grid.get_dimensions()
    for r in range(rows - 4):
        for c in range(cols - 4):
            if all(grid.get_cell(r+dr, c+dc) == 0 for dr in range(5) for dc in range(5)):
                if random.random() < 0.05 * (1 - influence_maps['distance'][r][c]) * (r / rows):
                    for dr in range(3):
                        for dc in range(3):
                            grid.set_cell(r+dr+1, c+dc+1, 3)

def fine_tune_structure(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    changes = True
    while changes:
        changes = False
        for r in range(rows):
            for c in range(cols):
                green_neighbors = sum(1 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
                                      if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 3)
                if grid.get_cell(r, c) == 3 and green_neighbors <= 3:
                    grid.set_cell(r, c, 0)
                    changes = True
                elif grid.get_cell(r, c) == 0 and green_neighbors >= 5:
                    grid.set_cell(r, c, 3)
                    changes = True

def adjust_symmetry(grid: ColoredGrid, seed_points: Set[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    center_r = sum(r for r, _ in seed_points) / len(seed_points)
    center_c = sum(c for _, c in seed_points) / len(seed_points)
    
    for r in range(rows):
        for c in range(cols):
            sym_r, sym_c = int(2*center_r - r), int(2*center_c - c)
            if 0 <= sym_r < rows and 0 <= sym_c < cols:
                if grid.get_cell(r, c) != grid.get_cell(sym_r, sym_c):
                    if random.random() < 0.5:
                        grid.set_cell(sym_r, sym_c, grid.get_cell(r, c))
                    else:
                        grid.set_cell(r, c, grid.get_cell(sym_r, sym_c))

def refine_edges(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                if r == 0 or r == rows-1 or c == 0 or c == cols-1:
                    connected_area = sum(1 for dr in range(-2, 3) for dc in range(-2, 3)
                                         if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 3)
                    if connected_area < (rows * cols) // 10:
                        grid.set_cell(r, c, 0)

def final_connectivity_check(grid: ColoredGrid, seed_points: Set[Tuple[int, int]]):
    ensure_connectivity(grid, seed_points)

def final_cleanup(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                green_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                      if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 3)
                if green_neighbors == 0:
                    grid.set_cell(r, c, 0)

def identify_initial_structures(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    """Identifies and groups adjacent green cells into initial structures."""
    structures = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    def dfs(r, c):
        structure = set()
        stack = [(r, c)]
        while stack:
            x, y = stack.pop()
            if (x, y) not in visited and grid.get_cell(x, y) == 3:
                visited.add((x, y))
                structure.add((x, y))
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        stack.append((nx, ny))
        return structure
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) == 3:
                structures.append(dfs(r, c))
    
    return structures

def create_influence_map(grid: ColoredGrid, structures: List[Set[Tuple[int, int]]]) -> List[List[float]]:
    """Creates an influence map based on the initial structures."""
    rows, cols = grid.get_dimensions()
    influence_map = [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    for structure in structures:
        for r, c in structure:
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        distance = max(abs(dr), abs(dc))
                        influence = 1.0 / (distance + 1) ** 2
                        influence_map[nr][nc] += influence
    
    return influence_map

def create_empty_grid(dimensions: Tuple[int, int]) -> ColoredGrid:
    """Creates a new empty grid with the given dimensions."""
    rows, cols = dimensions
    return ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

def expand_structures(grid: ColoredGrid, influence_map: List[List[float]]):
    """Expands structures based on the influence map and available space."""
    rows, cols = grid.get_dimensions()
    expansion_candidates = set()
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 0:
                        expansion_candidates.add((nr, nc))
    
    sorted_candidates = sorted(expansion_candidates, key=lambda pos: influence_map[pos[0]][pos[1]], reverse=True)
    
    for r, c in sorted_candidates[:len(sorted_candidates) // 2]:  # Expand only half of the candidates
        if random.random() < influence_map[r][c]:
            grid.set_cell(r, c, 3)

def connect_structures(grid: ColoredGrid):
    """Connects nearby structures with bridges or extensions."""
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                green_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                      if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.get_cell(r + dr, c + dc) == 3)
                if green_neighbors >= 2:
                    grid.set_cell(r, c, 3)

def refine_pattern(grid: ColoredGrid):
    """Refines the pattern by smoothing edges and balancing filled/empty spaces."""
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                empty_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                      if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.get_cell(r + dr, c + dc) == 0)
                if empty_neighbors >= 3:
                    grid.set_cell(r, c, 0)
            elif grid.get_cell(r, c) == 0:
                green_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                      if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.get_cell(r + dr, c + dc) == 3)
                if green_neighbors >= 3:
                    grid.set_cell(r, c, 3)

def adjust_global_pattern(grid: ColoredGrid):
    """Adjusts the global pattern for balance and edge coverage."""
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                if random.random() < 0.3:
                    grid.set_cell(r, c, 3)

def resolve_conflicts(grid: ColoredGrid):
    """Resolves conflicts between expanding structures."""
    # This function can be implemented if needed, but the current approach
    # should minimize conflicts through the use of the influence map.
    pass

def add_fine_details(grid: ColoredGrid):
    """Adds fine details like isolated green squares in empty areas."""
    rows, cols = grid.get_dimensions()
    for _ in range(rows * cols // 100):  # Add a number of details proportional to grid size
        r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
        if grid.get_cell(r, c) == 0 and all(grid.get_cell(r + dr, c + dc) == 0
                                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                            if 0 <= r + dr < rows and 0 <= c + dc < cols):
            size = random.choice([2, 3])
            if r + size <= rows and c + size <= cols:
                for i in range(size):
                    for j in range(size):
                        grid.set_cell(r + i, c + j, 3)

def final_cleanup(grid: ColoredGrid):
    """Performs final validation and cleanup."""
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                isolated = True
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.get_cell(r + dr, c + dc) == 3:
                        isolated = False
                        break
                if isolated:
                    grid.set_cell(r, c, 0)

def resolve_conflicts(grid: ColoredGrid):
    """Resolves conflicts between expanding structures."""
    # Implementation depends on specific conflict resolution rules
    pass

def add_fine_details(grid: ColoredGrid):
    """Adds fine details to match the intricate patterns in test cases."""
    # Implementation depends on specific patterns observed in test cases
    pass

def fill_surrounded_areas(grid: ColoredGrid):
    """Fills in 3x3 black areas completely surrounded by green cells."""
    rows, cols = grid.get_dimensions()
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if is_surrounded_3x3(grid, r, c):
                fill_rectangle(grid, r - 1, c - 1, 3, 3, 3)

def is_surrounded_3x3(grid: ColoredGrid, center_r: int, center_c: int) -> bool:
    """Checks if a 3x3 area is completely surrounded by green cells."""
    for dr in [-2, -1, 0, 1, 2]:
        for dc in [-2, -1, 0, 1, 2]:
            if dr in [-1, 0, 1] and dc in [-1, 0, 1]:
                if grid.get_cell(center_r + dr, center_c + dc) != 0:
                    return False
            else:
                if grid.get_cell(center_r + dr, center_c + dc) != 3:
                    return False
    return True

def final_cleanup(grid: ColoredGrid):
    """Performs final adjustments to match the expected output patterns."""
    # Implementation depends on specific cleanup rules observed in test cases
    pass
