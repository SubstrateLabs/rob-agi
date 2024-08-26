from rob_agi.colored_grid import ColoredGrid

def solve_cad67732(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by doubling its size and extending the pattern.
    
    The function:
    1. Creates a new grid double the size of the input.
    2. Copies the input pattern to the top-left quadrant.
    3. Expands the pattern horizontally and vertically.
    4. Fills in gaps based on surrounding colors.
    5. Iteratively completes the pattern until no changes are made.
    
    This approach preserves and extends existing patterns, handling
    diagonal, checkerboard, and other complex arrangements.
    """
    n, m = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(2*m)] for _ in range(2*n)])
    
    # Copy input grid to top-left quadrant
    for i in range(n):
        for j in range(m):
            output_grid.values[i][j] = input_grid.values[i][j]
    
    # Expand horizontally
    for i in range(2*n):
        for j in range(m):
            if output_grid.values[i][j] != 0:
                output_grid.values[i][j+m] = output_grid.values[i][j]
    
    # Expand vertically
    for i in range(n):
        for j in range(2*m):
            if output_grid.values[i][j] != 0:
                output_grid.values[i+n][j] = output_grid.values[i][j]
    
    # Fill in gaps
    max_iterations = 10
    for _ in range(max_iterations):
        changes_made = False
        for i in range(2*n):
            for j in range(2*m):
                if output_grid.values[i][j] == 0:
                    neighbors = []
                    for di, dj in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 2*n and 0 <= nj < 2*m:
                            neighbors.append(output_grid.values[ni][nj])
                    non_zero_neighbors = [x for x in neighbors if x != 0]
                    if non_zero_neighbors:
                        output_grid.values[i][j] = non_zero_neighbors[0]
                        changes_made = True
        if not changes_made:
            break
    
    return output_grid
