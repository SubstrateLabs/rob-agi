from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_9c56f360(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving green (3) squares to form 2x2 blocks
    while maximizing contact with sky blue (8) squares and moving towards
    the top-left when possible.

    1. Creates a deep copy of the input grid.
    2. Identifies all green squares and sky blue squares in the grid.
    3. Calculates sky blue contact scores for each position.
    4. Identifies and moves 2x2 green formations to optimal positions.
    5. Handles remaining individual green squares.
    6. Repeats the process until no more moves are possible.
    7. Optimizes by potentially splitting 2x2 formations for better contact.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with green squares moved and optimized.
    """
    grid = input_grid.deep_copy()
    changed = True
    while changed:
        changed = False
        green_squares = find_color_squares(grid, 3)
        sky_blue_squares = find_color_squares(grid, 8)
        contact_scores = calculate_contact_scores(grid, sky_blue_squares)
        
        # Handle 2x2 formations
        formations = find_2x2_formations(grid, green_squares)
        for formation in formations:
            if move_formation(grid, formation, contact_scores):
                changed = True
        
        # Handle remaining individual squares
        remaining_squares = [sq for sq in green_squares if not any(sq in f for f in formations)]
        for square in remaining_squares:
            if move_individual_square(grid, square, contact_scores):
                changed = True
    
    # Final optimization step
    optimize_formations(grid)
    
    return grid

def find_color_squares(grid: ColoredGrid, color: int) -> List[Tuple[int, int]]:
    squares = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == color:
                squares.append((r, c))
    return squares

def calculate_contact_scores(grid: ColoredGrid, sky_blue_squares: List[Tuple[int, int]]) -> List[List[int]]:
    rows, cols = grid.get_dimensions()
    scores = [[0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            scores[r][c] = sum(1 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                               if (r+dr, c+dc) in sky_blue_squares)
    return scores

def find_2x2_formations(grid: ColoredGrid, green_squares: List[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
    formations = []
    for r, c in green_squares:
        if all((r+dr, c+dc) in green_squares for dr in [0, 1] for dc in [0, 1]):
            formations.append({(r+dr, c+dc) for dr in [0, 1] for dc in [0, 1]})
    return formations

def move_formation(grid: ColoredGrid, formation: Set[Tuple[int, int]], contact_scores: List[List[int]]) -> bool:
    rows, cols = grid.get_dimensions()
    best_score = sum(contact_scores[r][c] for r, c in formation)
    best_pos = None
    
    for r in range(rows-1):
        for c in range(cols-1):
            if all(grid.get_cell(r+dr, c+dc) in [0, 3] for dr in [0, 1] for dc in [0, 1]):
                score = sum(contact_scores[r+dr][c+dc] for dr in [0, 1] for dc in [0, 1])
                if score > best_score or (score == best_score and (r, c) < best_pos):
                    best_score = score
                    best_pos = (r, c)
    
    if best_pos and best_pos != min(formation):
        for r, c in formation:
            grid.set_cell(r, c, 0)
        for dr in [0, 1]:
            for dc in [0, 1]:
                grid.set_cell(best_pos[0]+dr, best_pos[1]+dc, 3)
        return True
    return False

def move_individual_square(grid: ColoredGrid, pos: Tuple[int, int], contact_scores: List[List[int]]) -> bool:
    r, c = pos
    rows, cols = grid.get_dimensions()
    best_score = contact_scores[r][c]
    best_pos = pos
    
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 0:
                score = contact_scores[nr][nc]
                if score > best_score or (score == best_score and (nr, nc) < best_pos):
                    best_score = score
                    best_pos = (nr, nc)
    
    if best_pos != pos:
        grid.set_cell(r, c, 0)
        grid.set_cell(best_pos[0], best_pos[1], 3)
        return True
    return False

def optimize_formations(grid: ColoredGrid):
    green_squares = find_color_squares(grid, 3)
    sky_blue_squares = find_color_squares(grid, 8)
    contact_scores = calculate_contact_scores(grid, sky_blue_squares)
    formations = find_2x2_formations(grid, green_squares)
    
    for formation in formations:
        current_score = sum(contact_scores[r][c] for r, c in formation)
        best_split_score = 0
        best_split_positions = None
        
        for sq1, sq2 in [(sq1, sq2) for sq1 in formation for sq2 in formation if sq1 != sq2]:
            for dr1 in [-1, 0, 1]:
                for dc1 in [-1, 0, 1]:
                    for dr2 in [-1, 0, 1]:
                        for dc2 in [-1, 0, 1]:
                            nr1, nc1 = sq1[0] + dr1, sq1[1] + dc1
                            nr2, nc2 = sq2[0] + dr2, sq2[1] + dc2
                            if (is_valid_position(grid, nr1, nc1) and
                                is_valid_position(grid, nr2, nc2) and
                                grid.get_cell(nr1, nc1) == 0 and
                                grid.get_cell(nr2, nc2) == 0):
                                split_score = (contact_scores[nr1][nc1] +
                                               contact_scores[nr2][nc2] +
                                               sum(contact_scores[r][c] for r, c in formation if (r, c) not in {sq1, sq2}))
                                if split_score > best_split_score:
                                    best_split_score = split_score
                                    best_split_positions = ((sq1, (nr1, nc1)), (sq2, (nr2, nc2)))
        
        if best_split_score > current_score:
            for old_pos, new_pos in best_split_positions:
                grid.set_cell(old_pos[0], old_pos[1], 0)
                grid.set_cell(new_pos[0], new_pos[1], 3)

def is_valid_position(grid: ColoredGrid, row: int, col: int) -> bool:
    rows, cols = grid.get_dimensions()
    return 0 <= row < rows and 0 <= col < cols
