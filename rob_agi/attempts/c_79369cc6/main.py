from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_79369cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding or creating a yellow-magenta formation.
    
    The function identifies the best area for transformation based on existing
    magenta squares and yellow-magenta formations. It then expands this area
    by adding yellow squares above and to the left of magenta squares, creating
    a larger, connected yellow-magenta formation. Only one such transformation
    is applied per grid, and the rest of the grid remains unchanged.
    """
    def find_magenta_squares(grid: ColoredGrid) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid[r][c] == 6]

    def find_yellow_magenta_formations(grid: ColoredGrid) -> List[Tuple[int, int]]:
        formations = []
        visited = set()
        for r in range(grid.num_rows):
            for c in range(grid.num_cols):
                if (r, c) not in visited and grid[r][c] in [4, 6]:
                    formation = grid.find_connected_regions(grid[r][c])[0]
                    formations.append(formation[0])  # Use the first cell as the center
                    visited.update(formation)
        return formations

    def evaluate_area(grid: ColoredGrid, center: Tuple[int, int], size: int) -> float:
        r, c = center
        half = size // 2
        score = 0
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                    if grid[nr][nc] == 6:
                        score += 1
                    elif grid[nr][nc] == 4:
                        score += 0.5
        return score

    def design_formation(grid: ColoredGrid, center: Tuple[int, int], size: int) -> List[Tuple[int, int, int]]:
        r, c = center
        half = size // 2
        formation = []
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                    if grid[nr][nc] == 6:
                        formation.append((nr, nc, 6))
                    elif dr <= 0 or dc <= 0:  # Add yellow above and to the left
                        formation.append((nr, nc, 4))
        return formation

    def apply_formation(grid: ColoredGrid, formation: List[Tuple[int, int, int]]):
        for r, c, color in formation:
            grid.set_cell(r, c, color)

    magenta_squares = find_magenta_squares(input_grid)
    existing_formations = find_yellow_magenta_formations(input_grid)
    
    potential_areas = []
    for center in magenta_squares + existing_formations:
        score = evaluate_area(input_grid, center, size=7)
        potential_areas.append((center, score))
    
    best_area = max(potential_areas, key=lambda x: x[1])
    
    new_formation = design_formation(input_grid, best_area[0], size=7)
    
    output_grid = input_grid.deep_copy()
    apply_formation(output_grid, new_formation)
    
    return output_grid
