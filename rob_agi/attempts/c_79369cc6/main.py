from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_79369cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a new yellow-magenta formation in the quadrant that results in the most balanced distribution.
    
    The function divides the grid into four quadrants and analyzes each for the presence of yellow (4) and magenta (6) cells.
    It simulates adding a new formation (either 2x2 or 3x3) to each quadrant and chooses the one that results in the most balanced distribution.
    The new formation is placed in the corner of the chosen quadrant closest to the grid center.
    The formation consists of yellow (4) and magenta (6) cells in a specific pattern.
    Only one such transformation is applied per grid, and the rest of the grid remains unchanged.
    """
    def divide_into_quadrants(grid):
        rows, cols = grid.get_dimensions()
        mid_row, mid_col = rows // 2, cols // 2
        return [
            grid.extract_subgrid(0, 0, mid_row, mid_col),
            grid.extract_subgrid(0, mid_col, mid_row, cols - mid_col),
            grid.extract_subgrid(mid_row, 0, rows - mid_row, mid_col),
            grid.extract_subgrid(mid_row, mid_col, rows - mid_row, cols - mid_col)
        ]

    def count_yellow_magenta(quadrant):
        return sum(cell in [4, 6] for row in quadrant.values for cell in row)

    def get_formation_space(quadrant):
        rows, cols = quadrant.get_dimensions()
        return 3 if rows >= 3 and cols >= 3 else 2 if rows >= 2 and cols >= 2 else 0

    def simulate_addition(grid, quadrant_index, formation_size):
        quadrants = divide_into_quadrants(grid)
        quadrants[quadrant_index] = add_formation(quadrants[quadrant_index], formation_size)
        return [count_yellow_magenta(q) for q in quadrants]

    def add_formation(quadrant, size):
        new_quadrant = quadrant.deep_copy()
        rows, cols = new_quadrant.get_dimensions()
        if size == 3:
            new_quadrant.set_cell(rows - 3, cols - 3, 4)
            new_quadrant.set_cell(rows - 3, cols - 2, 4)
            new_quadrant.set_cell(rows - 2, cols - 3, 4)
            new_quadrant.set_cell(rows - 1, cols - 1, 6)
        elif size == 2:
            new_quadrant.set_cell(rows - 2, cols - 2, 4)
            new_quadrant.set_cell(rows - 1, cols - 1, 6)
        return new_quadrant

    def get_corner_position(quadrant_index, grid_size):
        rows, cols = grid_size
        if quadrant_index == 0: return (0, 0)
        elif quadrant_index == 1: return (0, cols // 2)
        elif quadrant_index == 2: return (rows // 2, 0)
        else: return (rows // 2, cols // 2)

    # Analyze quadrants
    quadrants = divide_into_quadrants(input_grid)
    spaces = [get_formation_space(q) for q in quadrants]

    # Simulate additions and calculate balance
    balance_scores = []
    for i, space in enumerate(spaces):
        if space > 0:
            distribution = simulate_addition(input_grid, i, space)
            balance_score = max(distribution) - min(distribution)
            balance_scores.append((balance_score, -i))  # Negative i for consistent tie-breaking
        else:
            balance_scores.append((float('inf'), -i))

    # Select target quadrant
    target_quad = min(range(4), key=lambda i: balance_scores[i])

    # Determine formation size and position
    size = spaces[target_quad]
    row, col = get_corner_position(target_quad, input_grid.get_dimensions())

    # Create new grid and apply transformation
    new_grid = input_grid.deep_copy()
    new_grid.set_cell(row, col, 4)  # Yellow
    new_grid.set_cell(row + size - 1, col + size - 1, 6)  # Magenta
    if size == 3:
        new_grid.set_cell(row, col + 1, 4)  # Additional Yellow for 3x3
        new_grid.set_cell(row + 1, col, 4)  # Additional Yellow for 3x3

    return new_grid
