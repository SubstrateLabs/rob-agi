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
    4. Arranges the rectangles in a new grid, optimizing for compactness
    5. Returns the new compact grid containing only the extracted regions

    The arrangement step now sorts regions by area and uses a more sophisticated
    packing algorithm to create a more compact output.

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

        # Sort regions by area in descending order
        sorted_regions = sorted(regions, key=lambda r: r[2] * r[3], reverse=True)

        max_width = max(r[3] for r in sorted_regions)
        total_area = sum(r[2] * r[3] for r in sorted_regions)
        initial_height = max(total_area // max_width, max(r[2] for r in sorted_regions))

        def can_place(grid, region, x, y):
            height, width = region[2], region[3]
            if y + height > len(grid) or x + width > len(grid[0]):
                return False
            return all(grid[y+i][x+j] == 0 for i in range(height) for j in range(width))

        def place_region(grid, region, x, y):
            top, left, height, width = region
            subgrid = input_grid.extract_subgrid(top, left, height, width)
            for i in range(height):
                for j in range(width):
                    grid[y+i][x+j] = subgrid.values[i][j]

        while True:
            grid = [[0 for _ in range(max_width)] for _ in range(initial_height)]
            placed = True

            for region in sorted_regions:
                placed = False
                for y in range(len(grid)):
                    for x in range(len(grid[0])):
                        if can_place(grid, region, x, y):
                            place_region(grid, region, x, y)
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    break

            if placed:
                # Trim empty rows and columns
                while grid and all(cell == 0 for cell in grid[-1]):
                    grid.pop()
                while grid and all(row[-1] == 0 for row in grid):
                    for row in grid:
                        row.pop()
                return ColoredGrid(values=grid)

            initial_height += 1

    bg_color = find_background_color(input_grid)
    regions = find_regions(input_grid, bg_color)
    return pack_regions(regions, input_grid)
