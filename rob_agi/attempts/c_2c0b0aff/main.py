from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2c0b0aff(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 2c0b0aff challenge by reconstructing a complete pattern from partial views.
    
    The function performs the following steps:
    1. Extracts non-black regions from the input grid
    2. Analyzes pattern pieces and their edge patterns
    3. Reconstructs the complete pattern by matching edge patterns
    4. Fills gaps and optimizes the solution
    5. Generates a compact output grid containing the reconstructed pattern
    
    Args:
    input_grid (ColoredGrid): The input grid containing partial pattern information
    
    Returns:
    ColoredGrid: A compact grid containing the reconstructed complete pattern
    """
    # Step 1: Extract non-black regions
    regions = extract_regions(input_grid)
    
    # Step 2: Analyze pattern pieces
    pieces = analyze_pieces(regions)
    
    # Step 3 & 4: Reconstruct pattern and fill gaps
    reconstructed = reconstruct_pattern(pieces)
    
    # Step 5: Generate output
    output = generate_output(reconstructed)
    
    return output

def extract_regions(grid: ColoredGrid) -> List[ColoredGrid]:
    regions = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) != 0:
                region = extract_region(grid, r, c, visited)
                regions.append(region)
    
    return regions

def extract_region(grid: ColoredGrid, start_r: int, start_c: int, visited: set) -> ColoredGrid:
    queue = [(start_r, start_c)]
    region = []
    min_r, min_c, max_r, max_c = start_r, start_c, start_r, start_c
    
    while queue:
        r, c = queue.pop(0)
        if (r, c) in visited or grid.get_cell(r, c) == 0:
            continue
        
        visited.add((r, c))
        region.append((r, c, grid.get_cell(r, c)))
        min_r, min_c = min(min_r, r), min(min_c, c)
        max_r, max_c = max(max_r, r), max(max_c, c)
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                queue.append((nr, nc))
    
    width, height = max_c - min_c + 1, max_r - min_r + 1
    region_grid = [[0 for _ in range(width)] for _ in range(height)]
    
    for r, c, val in region:
        region_grid[r - min_r][c - min_c] = val
    
    return ColoredGrid(values=region_grid)

def analyze_pieces(regions: List[ColoredGrid]) -> List[dict]:
    pieces = []
    for region in regions:
        piece = {
            'grid': region,
            'edges': get_edge_patterns(region),
        }
        pieces.append(piece)
    return pieces

def get_edge_patterns(grid: ColoredGrid) -> dict:
    rows, cols = grid.get_dimensions()
    return {
        'top': tuple(grid.get_cell(0, c) for c in range(cols)),
        'bottom': tuple(grid.get_cell(rows-1, c) for c in range(cols)),
        'left': tuple(grid.get_cell(r, 0) for r in range(rows)),
        'right': tuple(grid.get_cell(r, cols-1) for r in range(rows)),
    }

def reconstruct_pattern(pieces: List[dict]) -> ColoredGrid:
    # Start with the largest piece
    pieces.sort(key=lambda p: p['grid'].num_rows * p['grid'].num_cols, reverse=True)
    reconstructed = pieces[0]['grid']
    used_pieces = {0}
    
    while len(used_pieces) < len(pieces):
        best_match = None
        best_score = -1
        
        for i, piece in enumerate(pieces):
            if i in used_pieces:
                continue
            
            for edge in ['top', 'bottom', 'left', 'right']:
                score, position = match_edge(reconstructed, piece, edge)
                if score > best_score:
                    best_score = score
                    best_match = (i, position)
        
        if best_match:
            i, (r, c) = best_match
            reconstructed = add_piece(reconstructed, pieces[i]['grid'], r, c)
            used_pieces.add(i)
        else:
            break  # No more matches found
    
    return reconstructed

def match_edge(grid: ColoredGrid, piece: dict, edge: str) -> Tuple[int, Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    p_rows, p_cols = piece['grid'].get_dimensions()
    
    best_score = -1
    best_position = (-1, -1)
    
    if edge in ['top', 'bottom']:
        r = 0 if edge == 'top' else rows
        for c in range(cols - p_cols + 1):
            score = sum(grid.get_cell(r, c+i) == piece['grid'].get_cell(0, i) for i in range(p_cols))
            if score > best_score:
                best_score = score
                best_position = (r - p_rows if edge == 'top' else r, c)
    else:  # left or right
        c = 0 if edge == 'left' else cols
        for r in range(rows - p_rows + 1):
            score = sum(grid.get_cell(r+i, c) == piece['grid'].get_cell(i, 0) for i in range(p_rows))
            if score > best_score:
                best_score = score
                best_position = (r, c - p_cols if edge == 'left' else c)
    
    return best_score, best_position

def add_piece(grid: ColoredGrid, piece: ColoredGrid, r: int, c: int) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    p_rows, p_cols = piece.get_dimensions()
    
    new_rows = max(rows, r + p_rows)
    new_cols = max(cols, c + p_cols)
    
    new_grid = [[0 for _ in range(new_cols)] for _ in range(new_rows)]
    
    for i in range(rows):
        for j in range(cols):
            new_grid[i][j] = grid.get_cell(i, j)
    
    for i in range(p_rows):
        for j in range(p_cols):
            if piece.get_cell(i, j) != 0:
                new_grid[r+i][c+j] = piece.get_cell(i, j)
    
    return ColoredGrid(values=new_grid)

def generate_output(grid: ColoredGrid) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    min_r, min_c, max_r, max_c = rows, cols, 0, 0
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 0:
                min_r, min_c = min(min_r, r), min(min_c, c)
                max_r, max_c = max(max_r, r), max(max_c, c)
    
    output = [[grid.get_cell(r, c) for c in range(min_c, max_c+1)] for r in range(min_r, max_r+1)]
    return ColoredGrid(values=output)
