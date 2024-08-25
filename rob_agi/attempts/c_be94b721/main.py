from rob_agi.colored_grid import ColoredGrid

def solve_be94b721(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the be94b721 challenge by finding the largest non-zero color region in the input grid
    and extracting it as a new ColoredGrid.

    The function performs the following steps:
    1. Analyze the input grid to identify all non-zero color regions.
    2. For each non-zero cell, perform a flood fill to find the connected region.
    3. Keep track of the largest region found, storing its bounding box coordinates.
    4. Extract the largest region using the bounding box coordinates.
    5. Return the extracted region as a new ColoredGrid.

    Args:
        input_grid (ColoredGrid): The input grid to process.

    Returns:
        ColoredGrid: The largest non-zero color region extracted from the input grid.
    """
    def flood_fill(row, col, color, visited):
        if (row < 0 or row >= len(input_grid.values) or
            col < 0 or col >= len(input_grid.values[0]) or
            input_grid.values[row][col] != color or
            (row, col) in visited):
            return set()
        
        visited.add((row, col))
        region = {(row, col)}
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            region.update(flood_fill(row + dr, col + dc, color, visited))
        return region

    largest_region = set()
    visited = set()

    for r in range(len(input_grid.values)):
        for c in range(len(input_grid.values[0])):
            if input_grid.values[r][c] != 0 and (r, c) not in visited:
                region = flood_fill(r, c, input_grid.values[r][c], visited)
                if len(region) > len(largest_region):
                    largest_region = region

    if not largest_region:
        return ColoredGrid(values=[[0]])

    min_row = min(r for r, _ in largest_region)
    max_row = max(r for r, _ in largest_region)
    min_col = min(c for _, c in largest_region)
    max_col = max(c for _, c in largest_region)

    result = []
    for r in range(min_row, max_row + 1):
        row = []
        for c in range(min_col, max_col + 1):
            row.append(input_grid.values[r][c])
        result.append(row)

    return ColoredGrid(values=result)
