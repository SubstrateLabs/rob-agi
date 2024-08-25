from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7c9b52a0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying non-background rectangular regions,
    extracting them, and arranging them in a new compact grid.

    1. Identifies the background color
    2. Finds rectangular regions of non-background colors
    3. Extracts these regions
    4. Arranges the regions in a new grid, preserving their order and aligning them to the left
    5. Returns the new compact grid containing only the extracted regions

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: A new grid containing the extracted and arranged regions
    """
    def find_background_color(grid: ColoredGrid) -> int:
        rows, cols = grid.get_dimensions()
        border = (
            grid.values[0] + grid.values[-1] +
            [row[0] for row in grid.values[1:-1]] +
            [row[-1] for row in grid.values[1:-1]]
        )
        return max(set(border), key=border.count)

    def find_regions(grid: ColoredGrid, bg_color: int) -> List[Tuple[int, int, int, int]]:
        rows, cols = grid.get_dimensions()
        regions = []
        visited = set()

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.values[r][c] != bg_color:
                    width = height = 1
                    while c + width < cols and grid.values[r][c + width] != bg_color:
                        width += 1
                    while r + height < rows and all(grid.values[r + height][c + i] != bg_color for i in range(width)):
                        height += 1
                    regions.append((r, c, height, width))
                    for i in range(height):
                        for j in range(width):
                            visited.add((r + i, c + j))

        return regions

    def arrange_regions(regions: List[Tuple[int, int, int, int]], input_grid: ColoredGrid) -> ColoredGrid:
        max_width = max(region[3] for region in regions)
        total_height = sum(region[2] for region in regions)
        output = [[0 for _ in range(max_width)] for _ in range(total_height)]

        current_y = 0
        for top, left, height, width in regions:
            subgrid = input_grid.extract_subgrid(top, left, height, width)
            for i in range(height):
                for j in range(width):
                    output[current_y + i][j] = subgrid.values[i][j]
            current_y += height

        return ColoredGrid(values=output)

    bg_color = find_background_color(input_grid)
    regions = find_regions(input_grid, bg_color)
    return arrange_regions(regions, input_grid)
