from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_15113be4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by enhancing or introducing a secondary color (green, magenta, or sky blue)
    in a balanced and visually interesting pattern across all quadrants. The function follows these steps:
    1. Identifies the secondary color to use (3: green, 6: magenta, or 8: sky blue).
    2. Analyzes the existing pattern, distribution, and complexity of colors in each quadrant.
    3. Creates a pattern strategy based on the analysis.
    4. Enhances existing secondary color areas by forming geometric patterns like L-shapes, clusters, or diagonal paths.
    5. Introduces new instances of the secondary color following the pattern strategy.
    6. Balances the distribution of the secondary color across quadrants.
    7. Refines the transformation for consistency and visual appeal.
    8. Preserves the yellow (4) grid structure throughout the process.
    9. Interacts with existing colors, especially blue (1), to create visually interesting patterns.

    The transformation aims to create a balanced, aesthetically pleasing distribution of the secondary color
    while maintaining the original grid's structure and enhancing existing patterns.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the applied pattern.
    """
    output_grid = input_grid.deep_copy()
    secondary_color = identify_secondary_color(output_grid)
    
    analysis = analyze_grid(output_grid, secondary_color)
    strategy = create_pattern_strategy(analysis, secondary_color)
    
    enhance_existing_areas(output_grid, secondary_color, strategy)
    introduce_new_instances(output_grid, secondary_color, strategy)
    balance_distribution(output_grid, secondary_color, analysis)
    refine_transformation(output_grid, secondary_color)

    return output_grid

def identify_secondary_color(grid: ColoredGrid) -> int:
    colors = grid.get_unique_colors()
    if 3 in colors:
        return 3  # green
    elif 6 in colors:
        return 6  # magenta
    elif 8 in colors:
        return 8  # sky blue
    else:
        return 8  # default to sky blue if no secondary color is present

def analyze_grid(grid: ColoredGrid, color: int) -> Dict:
    rows, cols = grid.get_dimensions()
    quadrants = {1: [0, rows//2, 0, cols//2],
                 2: [0, rows//2, cols//2, cols],
                 3: [rows//2, rows, 0, cols//2],
                 4: [rows//2, rows, cols//2, cols]}
    
    analysis = {}
    for q, (r_start, r_end, c_start, c_end) in quadrants.items():
        quadrant_cells = [(r, c) for r in range(r_start, r_end) for c in range(c_start, c_end)]
        color_count = sum(1 for r, c in quadrant_cells if grid.get_cell(r, c) == color)
        blue_count = sum(1 for r, c in quadrant_cells if grid.get_cell(r, c) == 1)
        diagonal_paths = find_diagonal_paths(grid, color, quadrant_cells)
        clusters = find_clusters(grid, color, quadrant_cells)
        
        complexity_score = color_count + blue_count + len(diagonal_paths) * 2 + len(clusters) * 3
        
        analysis[q] = {
            'color_count': color_count,
            'blue_count': blue_count,
            'diagonal_paths': diagonal_paths,
            'clusters': clusters,
            'complexity_score': complexity_score
        }
    
    return analysis

def create_pattern_strategy(analysis: Dict, color: int) -> Dict:
    avg_complexity = sum(q['complexity_score'] for q in analysis.values()) / len(analysis)
    strategy = {}
    for q, data in analysis.items():
        if data['complexity_score'] < avg_complexity:
            strategy[q] = {
                'add_diagonals': len(data['diagonal_paths']) < 2,
                'enhance_clusters': len(data['clusters']) < 3,
                'target_increase': int(avg_complexity - data['complexity_score'])
            }
        else:
            strategy[q] = {
                'add_diagonals': False,
                'enhance_clusters': False,
                'target_increase': 0
            }
    return strategy

def enhance_existing_areas(grid: ColoredGrid, color: int, strategy: Dict):
    for q, strat in strategy.items():
        if strat['enhance_clusters']:
            enhance_clusters(grid, color, q)
        if strat['add_diagonals']:
            add_diagonal_paths(grid, color, q)

def introduce_new_instances(grid: ColoredGrid, color: int, strategy: Dict):
    rows, cols = grid.get_dimensions()
    quadrants = {1: [0, rows//2, 0, cols//2],
                 2: [0, rows//2, cols//2, cols],
                 3: [rows//2, rows, 0, cols//2],
                 4: [rows//2, rows, cols//2, cols]}
    
    for q, strat in strategy.items():
        r_start, r_end, c_start, c_end = quadrants[q]
        added = 0
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                if added >= strat['target_increase']:
                    break
                if is_valid_cell(grid, r, c) and grid.get_cell(r, c) in [0, 1]:
                    if not has_adjacent_color(grid, r, c, color):
                        grid.set_cell(r, c, color)
                        added += 1
            if added >= strat['target_increase']:
                break

def balance_distribution(grid: ColoredGrid, color: int, analysis: Dict):
    total_color = sum(data['color_count'] for data in analysis.values())
    target_per_quadrant = total_color // 4
    
    rows, cols = grid.get_dimensions()
    quadrants = {1: [0, rows//2, 0, cols//2],
                 2: [0, rows//2, cols//2, cols],
                 3: [rows//2, rows, 0, cols//2],
                 4: [rows//2, rows, cols//2, cols]}
    
    for q, data in analysis.items():
        diff = target_per_quadrant - data['color_count']
        r_start, r_end, c_start, c_end = quadrants[q]
        if diff > 0:
            # Add color
            for _ in range(diff):
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        if grid.get_cell(r, c) in [0, 1] and not has_adjacent_color(grid, r, c, color):
                            grid.set_cell(r, c, color)
                            break
                    else:
                        continue
                    break
        elif diff < 0:
            # Remove color
            for _ in range(-diff):
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        if grid.get_cell(r, c) == color and not is_critical_cell(grid, r, c, color):
                            grid.set_cell(r, c, 0)
                            break
                    else:
                        continue
                    break

def refine_transformation(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == color:
                if not has_adjacent_color(grid, r, c, color):
                    # Remove isolated color cells
                    grid.set_cell(r, c, 0)
                elif forms_l_shape(grid, r, c, color):
                    # Enhance L-shapes
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = r + dr, c + dc
                        if is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) in [0, 1]:
                            grid.set_cell(nr, nc, color)
                            break

def is_valid_cell(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    return 0 <= r < rows and 0 <= c < cols and grid.get_cell(r, c) != 4

def has_adjacent_color(grid: ColoredGrid, r: int, c: int, color: int) -> bool:
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) == color:
            return True
    return False

def forms_l_shape(grid: ColoredGrid, r: int, c: int, color: int) -> bool:
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i, (dr1, dc1) in enumerate(directions):
        for dr2, dc2 in directions[i+1:]:
            if (is_valid_cell(grid, r+dr1, c+dc1) and grid.get_cell(r+dr1, c+dc1) == color and
                is_valid_cell(grid, r+dr2, c+dc2) and grid.get_cell(r+dr2, c+dc2) == color):
                return True
    return False

def is_critical_cell(grid: ColoredGrid, r: int, c: int, color: int) -> bool:
    # A cell is critical if removing it would create an isolated color cell
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) == color:
            if not has_adjacent_color(grid, nr, nc, color, exclude=(r, c)):
                return True
    return False

def has_adjacent_color(grid: ColoredGrid, r: int, c: int, color: int, exclude: Tuple[int, int] = None) -> bool:
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if (nr, nc) != exclude and is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) == color:
            return True
    return False

def find_diagonal_paths(grid: ColoredGrid, color: int, cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    paths = []
    visited = set()
    for r, c in cells:
        if (r, c) not in visited and grid.get_cell(r, c) == color:
            path = []
            dr, dc = 1, 1
            while is_valid_cell(grid, r, c) and grid.get_cell(r, c) == color:
                path.append((r, c))
                visited.add((r, c))
                r, c = r + dr, c + dc
            if len(path) > 2:
                paths.append(path)
    return paths

def find_clusters(grid: ColoredGrid, color: int, cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    clusters = []
    visited = set()
    for r, c in cells:
        if (r, c) not in visited and grid.get_cell(r, c) == color:
            cluster = []
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if (cr, cc) not in visited and grid.get_cell(cr, cc) == color:
                    cluster.append((cr, cc))
                    visited.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (nr, nc) in cells and (nr, nc) not in visited:
                            stack.append((nr, nc))
            if len(cluster) > 1:
                clusters.append(cluster)
    return clusters

def enhance_clusters(grid: ColoredGrid, color: int, quadrant: int):
    rows, cols = grid.get_dimensions()
    quadrants = {1: [0, rows//2, 0, cols//2],
                 2: [0, rows//2, cols//2, cols],
                 3: [rows//2, rows, 0, cols//2],
                 4: [rows//2, rows, cols//2, cols]}
    r_start, r_end, c_start, c_end = quadrants[quadrant]
    
    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            if grid.get_cell(r, c) == color:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) in [0, 1]:
                        grid.set_cell(nr, nc, color)
                        break

def add_diagonal_paths(grid: ColoredGrid, color: int, quadrant: int):
    rows, cols = grid.get_dimensions()
    quadrants = {1: [0, rows//2, 0, cols//2],
                 2: [0, rows//2, cols//2, cols],
                 3: [rows//2, rows, 0, cols//2],
                 4: [rows//2, rows, cols//2, cols]}
    r_start, r_end, c_start, c_end = quadrants[quadrant]
    
    for r in range(r_start, r_end - 2):
        for c in range(c_start, c_end - 2):
            if all(grid.get_cell(r+i, c+i) in [0, 1] for i in range(3)):
                for i in range(3):
                    grid.set_cell(r+i, c+i, color)
                break
        else:
            continue
        break
