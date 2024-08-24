from rob_agi.colored_grid import ColoredGrid
import copy

def solve_88a10436(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 88a10436 challenge by moving a pattern to a new position centered around an anchor point.
    
    The function does the following:
    1. Locates the anchor point (cell with value 5) in the grid.
    2. Identifies the pattern by finding the bounding box of non-zero, non-5 values.
    3. Calculates the new position to center the pattern around the anchor point.
    4. Moves the pattern to the new position, ensuring it stays within grid boundaries.
    5. Clears the original pattern and anchor point.
    6. Handles edge cases (no anchor point or no pattern) by returning the original grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the pattern moved to the new position.
    """
    grid = input_grid.values
    rows, cols = len(grid), len(grid[0])

    # Find anchor point (5)
    anchor = next(((r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5), None)
    
    if not anchor:
        return input_grid  # No anchor point, return original grid

    # Find pattern bounds
    pattern = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] not in (0, 5)]
    
    if not pattern:
        return input_grid  # No pattern, return original grid

    min_r, min_c = min(r for r, _ in pattern), min(c for _, c in pattern)
    max_r, max_c = max(r for r, _ in pattern), max(c for _, c in pattern)
    
    pattern_height = max_r - min_r + 1
    pattern_width = max_c - min_c + 1

    # Calculate new position to center the pattern around the anchor point
    new_min_r = anchor[0] - (pattern_height // 2)
    new_min_c = anchor[1] - (pattern_width // 2)

    # Create a deep copy of the original grid to preserve it
    new_grid = copy.deepcopy(grid)

    # Move pattern to new position
    for r in range(pattern_height):
        for c in range(pattern_width):
            old_r, old_c = min_r + r, min_c + c
            new_r, new_c = new_min_r + r, new_min_c + c
            if 0 <= new_r < rows and 0 <= new_c < cols:
                new_grid[new_r][new_c] = grid[old_r][old_c]

    # Clear original pattern and anchor point
    for r, c in pattern:
        grid[r][c] = 0
    grid[anchor[0]][anchor[1]] = 0

    # Merge new pattern into original grid, preserving existing patterns
    for r in range(rows):
        for c in range(cols):
            if new_grid[r][c] != 0:
                grid[r][c] = new_grid[r][c]

    return ColoredGrid(values=grid)
