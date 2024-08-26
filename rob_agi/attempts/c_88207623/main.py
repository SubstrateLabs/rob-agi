from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_88207623(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by expanding single pixels adjacent to main shapes.
    
    1. Identifies main shapes and their border colors.
    2. Finds single pixels of unique colors adjacent to main shapes.
    3. Expands these pixels to form new borders around the main shapes.
    4. Preserves the original main shapes and their borders.
    
    The expansion stops at existing borders, other colors, or grid edges.
    """
    def find_main_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
        main_color = max(set(cell for row in grid.values for cell in row) - {0, 2}, key=lambda x: sum(row.count(x) for row in grid.values))
        return grid.find_connected_regions(main_color)

    def find_border_color(grid: ColoredGrid, shapes: List[List[Tuple[int, int]]]) -> int:
        for shape in shapes:
            for r, c in shape:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < len(grid.values) and 0 <= nc < len(grid.values[0]):
                        if grid.values[nr][nc] not in {0, grid.values[r][c]}:
                            return grid.values[nr][nc]
        return 2  # Default to red if no border found

    def find_expanding_pixels(grid: ColoredGrid, shapes: List[List[Tuple[int, int]]], border_color: int) -> List[Tuple[int, int, int]]:
        expanding_pixels = []
        shape_cells = set(cell for shape in shapes for cell in shape)
        for r in range(len(grid.values)):
            for c in range(len(grid.values[0])):
                if grid.values[r][c] not in {0, border_color} and (r, c) not in shape_cells:
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < len(grid.values) and 0 <= nc < len(grid.values[0]):
                            if (nr, nc) in shape_cells:
                                expanding_pixels.append((r, c, grid.values[r][c]))
                                break
        return expanding_pixels

    def expand_color(grid: ColoredGrid, start: Tuple[int, int], color: int, border_color: int, shapes: List[List[Tuple[int, int]]]) -> None:
        rows, cols = len(grid.values), len(grid.values[0])
        stack = [start]
        visited = set()
        shape_cells = set(cell for shape in shapes for cell in shape)

        while stack:
            r, c = stack.pop()
            if (r, c) in visited or grid.values[r][c] == border_color or (r, c) in shape_cells:
                continue
            
            grid.values[r][c] = color
            visited.add((r, c))

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 0:
                    stack.append((nr, nc))

    # Main solving process
    result_grid = input_grid.deep_copy()
    main_shapes = find_main_shapes(result_grid)
    border_color = find_border_color(result_grid, main_shapes)
    expanding_pixels = find_expanding_pixels(result_grid, main_shapes, border_color)

    for r, c, color in expanding_pixels:
        expand_color(result_grid, (r, c), color, border_color, main_shapes)

    # Preserve original shapes and borders
    for shape in main_shapes:
        for r, c in shape:
            result_grid.values[r][c] = input_grid.values[r][c]

    return result_grid
