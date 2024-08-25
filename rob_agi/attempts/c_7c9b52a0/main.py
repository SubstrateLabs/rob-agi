from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_7c9b52a0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying non-background rectangular regions,
    extracting them, and arranging them in a new compact grid.

    1. Identifies the background color
    2. Finds non-background color clusters using flood fill
    3. Extracts minimal rectangles containing these clusters
    4. Arranges the rectangles in a new grid, preserving their left-to-right order
       and packing them efficiently
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

    def flood_fill(grid: ColoredGrid, start_r: int, start_c: int, bg_color: int) -> Tuple[int, int, int, int]:
        rows, cols = grid.get_dimensions()
        color = grid.values[start_r][start_c]
        queue = deque([(start_r, start_c)])
        visited = set()
        min_r, min_c, max_r, max_c = start_r, start_c, start_r, start_c

        while queue:
            r, c = queue.popleft()
            if (r, c) in visited or grid.values[r][c] != color:
                continue
            visited.add((r, c))
            min_r, min_c = min(min_r, r), min(min_c, c)
            max_r, max_c = max(max_r, r), max(max_c, c)

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] != bg_color:
                    queue.append((nr, nc))

        return min_r, min_c, max_r - min_r + 1, max_c - min_c + 1

    def find_regions(grid: ColoredGrid, bg_color: int) -> List[Tuple[int, int, int, int]]:
        rows, cols = grid.get_dimensions()
        regions = []
        visited = set()

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.values[r][c] != bg_color:
                    region = flood_fill(grid, r, c, bg_color)
                    regions.append(region)
                    top, left, height, width = region
                    for i in range(height):
                        for j in range(width):
                            visited.add((top + i, left + j))

        return sorted(regions, key=lambda x: x[1])  # Sort by left coordinate

    def pack_regions(regions: List[Tuple[int, int, int, int]], input_grid: ColoredGrid) -> ColoredGrid:
        if not regions:
            return ColoredGrid(values=[[]])

        output = []
        current_row = []
        max_height = 0
        max_width = 0

        for top, left, height, width in regions:
            if current_row and sum(r[3] for r in current_row) + width > sum(r[2] for r in current_row):
                output.append(current_row)
                current_row = []
                max_height = 0

            current_row.append((top, left, height, width))
            max_height = max(max_height, height)
            max_width = max(max_width, sum(r[3] for r in current_row))

        if current_row:
            output.append(current_row)

        packed_grid = [[0 for _ in range(max_width)] for _ in range(sum(max(r[2] for r in row) for row in output))]

        y_offset = 0
        for row in output:
            x_offset = 0
            row_height = max(r[2] for r in row)
            for top, left, height, width in row:
                subgrid = input_grid.extract_subgrid(top, left, height, width)
                for i in range(height):
                    for j in range(width):
                        packed_grid[y_offset + i][x_offset + j] = subgrid.values[i][j]
                x_offset += width
            y_offset += row_height

        return ColoredGrid(values=packed_grid)

    bg_color = find_background_color(input_grid)
    regions = find_regions(input_grid, bg_color)
    return pack_regions(regions, input_grid)
