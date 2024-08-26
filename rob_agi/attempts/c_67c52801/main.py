from rob_agi.colored_grid import ColoredGrid

def solve_67c52801(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving colored cells downward while preserving their column positions and connected regions.
    
    The transformation follows these rules:
    1. The bottom row of the input grid remains unchanged.
    2. Colored cells move vertically downward, maintaining their original column positions and order.
    3. Connected regions of the same color maintain their shape and relative positions.
    4. Empty space (black/0) fills from the top down.
    
    The algorithm works as follows:
    1. Initialize the output grid with zeros and copy the bottom row from the input.
    2. Identify connected regions in the input grid.
    3. Sort regions by their bottom-most row (in descending order).
    4. Place each region in the output grid, starting from the bottom.
    5. Return the transformed grid.
    
    Returns:
    ColoredGrid: The transformed grid according to the specified rules.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Copy the bottom row
    output_grid.values[-1] = input_grid.values[-1].copy()
    
    # Identify connected regions
    visited = set()
    regions = []
    
    def dfs(r, c, color, region):
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or input_grid.values[r][c] != color:
            return
        visited.add((r, c))
        region.append((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(r + dr, c + dc, color, region)
    
    for r in range(rows - 1):  # Exclude bottom row
        for c in range(cols):
            if (r, c) not in visited and input_grid.values[r][c] != 0:
                region = []
                dfs(r, c, input_grid.values[r][c], region)
                regions.append(region)
    
    # Sort regions by their bottom-most row (in descending order)
    regions.sort(key=lambda x: max(r for r, _ in x), reverse=True)
    
    # Place regions in the output grid
    for region in regions:
        color = input_grid.values[region[0][0]][region[0][1]]
        bottom = max(r for r, _ in region)
        offset = rows - 2 - bottom  # How much to move the region down
        
        for r, c in region:
            new_r = r + offset
            if new_r < rows - 1:  # Ensure we don't overwrite the bottom row
                output_grid.values[new_r][c] = color
    
    return output_grid
