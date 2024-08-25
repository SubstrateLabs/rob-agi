from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7c9b52a0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying non-background rectangular regions,
    extracting them, and arranging them in a new compact grid.

    1. Identifies the background color
    2. Finds rectangular regions of non-background colors
    3. Extracts these regions
    4. Arranges the regions in a new grid, sorted by size
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

        return sorted(regions, key=lambda x: (-x[2], -x[3]))  # Sort by height, then width

    def arrange_regions(regions: List[Tuple[int, int, int, int]], input_grid: ColoredGrid) -> ColoredGrid:
        total_area = sum(region[2] * region[3] for region in regions)
        side = int(total_area ** 0.5) + 1
        output = [[0 for _ in range(side)] for _ in range(side)]

        def can_place(r, c, height, width):
            if r + height > side or c + width > side:
                return False
            return all(output[i][j] == 0 for i in range(r, r + height) for j in range(c, c + width))

        for top, left, height, width in regions:
            placed = False
            for r in range(side):
                for c in range(side):
                    if can_place(r, c, height, width):
                        subgrid = input_grid.extract_subgrid(top, left, height, width)
                        for i in range(height):
                            for j in range(width):
                                output[r + i][c + j] = subgrid.values[i][j]
                        placed = True
                        break
                if placed:
                    break

        # Trim empty rows and columns
        while all(cell == 0 for cell in output[-1]):
            output.pop()
        while all(row[-1] == 0 for row in output):
            for row in output:
                row.pop()

        return ColoredGrid(values=output)

    bg_color = find_background_color(input_grid)
    regions = find_regions(input_grid, bg_color)
    return arrange_regions(regions, input_grid)
