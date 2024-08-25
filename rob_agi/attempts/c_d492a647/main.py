from rob_agi.colored_grid import ColoredGrid

def find_color_to_add(grid):
    for row in grid.values:
        for cell in row:
            if cell in [1, 3]:  # Blue or Green
                return cell
    return 3  # Default to Green if no color found

def find_enclosed_regions(grid):
    def is_enclosed(r, c):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid.values) and 0 <= nc < len(grid.values[0]):
                if grid.values[nr][nc] == 0:
                    return False
            else:
                return False
        return True

    def flood_fill(r, c, region):
        if (r, c) in region or grid.values[r][c] != 0:
            return
        region.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid.values) and 0 <= nc < len(grid.values[0]):
                flood_fill(nr, nc, region)

    enclosed_regions = []
    visited = set()
    for r in range(len(grid.values)):
        for c in range(len(grid.values[0])):
            if grid.values[r][c] == 0 and (r, c) not in visited and is_enclosed(r, c):
                region = set()
                flood_fill(r, c, region)
                enclosed_regions.append(region)
                visited.update(region)

    return enclosed_regions

def solve_d492a647(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following steps:
    1. Determines the fill color (blue or green) based on the first non-black, non-gray color found in the input grid.
    2. Creates a deep copy of the input grid.
    3. Applies a global checkerboard pattern to the copied grid:
       - Fills black squares with the determined color where the sum of row and column indices is odd.
       - Preserves all non-black colors from the input.
    4. Returns the modified grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the checkerboard pattern applied.
    """
    # Step 1: Determine the fill color
    fill_color = 3  # Default to green
    for row in input_grid.values:
        for cell in row:
            if cell not in [0, 5]:  # If not black or gray
                fill_color = cell
                break
        if fill_color != 3:
            break

    # Step 2: Create a copy of the input grid
    output_grid = input_grid.deep_copy()

    # Step 3: Apply the global checkerboard pattern
    for r in range(len(output_grid.values)):
        for c in range(len(output_grid.values[r])):
            if output_grid.values[r][c] == 0 and (r + c) % 2 == 1:
                output_grid.values[r][c] = fill_color

    # Step 4: Return the modified grid
    return output_grid
