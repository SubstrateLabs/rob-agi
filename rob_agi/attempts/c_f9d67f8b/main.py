from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def analyze_pattern(grid: ColoredGrid, row: int, col: int) -> int:
    """Analyze the pattern around a cell and determine the most suitable color."""
    rows, cols = grid.get_dimensions()
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                color = grid.get_cell(nr, nc)
                if color != 9:  # Ignore brown cells
                    neighbors.append((color, abs(dr) + abs(dc)))  # Color and priority

    if not neighbors:
        return 0  # Default to black if no valid neighbors

    color_counts = Counter(color for color, _ in neighbors)
    max_count = max(color_counts.values())
    candidates = [color for color, count in color_counts.items() if count == max_count]

    if len(candidates) == 1:
        return candidates[0]

    # Tiebreaker based on priority (1 for orthogonal, 2 for diagonal)
    return min((color for color, priority in neighbors if color in candidates), key=lambda x: neighbors[neighbors.index((x, 1)) if (x, 1) in neighbors else neighbors.index((x, 2))])

def solve_f9d67f8b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by:
    1. Identifying all brown (9) cells.
    2. Analyzing the pattern around each brown cell.
    3. Replacing brown cells with the most suitable color based on surrounding patterns.
    4. Repeating the process until all brown cells are replaced.

    Args:
        input_grid (ColoredGrid): The input grid to transform.

    Returns:
        ColoredGrid: The transformed grid with brown cells replaced according to surrounding patterns.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()
    brown_cells = [(r, c) for r in range(rows) for c in range(cols) if new_grid.get_cell(r, c) == 9]

    while brown_cells:
        changes_made = False
        for r, c in brown_cells[:]:
            new_color = analyze_pattern(new_grid, r, c)
            if new_color != 9:
                new_grid.set_cell(r, c, new_color)
                brown_cells.remove((r, c))
                changes_made = True
        
        if not changes_made:
            # Handle any remaining brown cells
            for r, c in brown_cells:
                for d in range(1, max(rows, cols)):
                    for dr, dc in [(0, d), (d, 0), (0, -d), (-d, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and new_grid.get_cell(nr, nc) != 9:
                            new_grid.set_cell(r, c, new_grid.get_cell(nr, nc))
                            brown_cells.remove((r, c))
                            break
                    if (r, c) not in brown_cells:
                        break

    return new_grid
