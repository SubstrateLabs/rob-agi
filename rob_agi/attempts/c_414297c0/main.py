from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_414297c0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by selecting the most efficient background color,
    preserving all other colored elements and formations, and arranging them
    in a compact manner while maintaining their relative positions.
    
    1. Analyzes the input grid to identify all unique colors and their regions.
    2. Selects the background color that results in the smallest output grid.
    3. Identifies key formations and individual elements to preserve.
    4. Creates a new grid with the chosen background color.
    5. Places preserved elements and formations, optimizing for compactness.
    6. Ensures all colored elements touch either an edge or another colored element.
    7. Optimizes the grid by removing unnecessary background-only areas.
    
    Returns a new ColoredGrid object with the transformed grid.
    """
    # Step 1 & 2: Analyze input and select background color
    background_color = select_background_color(input_grid)
    
    # Step 3: Identify elements and formations to preserve
    elements_and_formations = find_elements_and_formations(input_grid, background_color)
    
    # Step 4 & 5: Create initial output grid and place elements
    output_grid = create_and_place_elements(elements_and_formations, background_color)
    
    # Step 6 & 7: Optimize and compact the grid
    optimized_grid = optimize_and_compact_grid(output_grid, background_color)
    
    # Create and return the final ColoredGrid object
    return ColoredGrid(values=optimized_grid)

def select_background_color(grid: ColoredGrid) -> int:
    """Selects the most efficient background color."""
    color_counts = grid.get_color_frequencies()
    max_count = max(color_counts.values())
    return max(color_counts, key=color_counts.get)

def find_elements_and_formations(grid: ColoredGrid, background_color: int) -> List[Tuple[int, List[Tuple[int, int]]]]:
    elements_and_formations = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != background_color and grid.values[r][c] != 0:
                color = grid.values[r][c]
                formation = []
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) not in visited and 0 <= cr < rows and 0 <= cc < cols and grid.values[cr][cc] == color:
                        visited.add((cr, cc))
                        formation.append((cr, c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            stack.append((cr + dr, cc + dc))
                elements_and_formations.append((color, formation))
    
    return elements_and_formations

def create_and_place_elements(elements_and_formations: List[Tuple[int, List[Tuple[int, int]]]], background_color: int) -> List[List[int]]:
    if not elements_and_formations:
        return [[background_color]]
    
    # Calculate the dimensions of the output grid
    all_coords = [coord for _, formation in elements_and_formations for coord in formation]
    min_row = min(r for r, _ in all_coords)
    max_row = max(r for r, _ in all_coords)
    min_col = min(c for _, c in all_coords)
    max_col = max(c for _, c in all_coords)
    
    height = max_row - min_row + 3  # Add some padding
    width = max_col - min_col + 3
    
    grid = [[background_color for _ in range(width)] for _ in range(height)]
    
    for color, formation in elements_and_formations:
        for r, c in formation:
            new_r, new_c = r - min_row + 1, c - min_col + 1
            if 0 <= new_r < height and 0 <= new_c < width:
                grid[new_r][new_c] = color
    
    return grid

def optimize_and_compact_grid(grid: List[List[int]], background_color: int) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])
    
    def is_touching(r: int, c: int) -> bool:
        if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
            return True
        return any(grid[r + dr][c + dc] != background_color
                   for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)])
    
    # Compact the grid
    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != background_color and not is_touching(r, c):
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and is_touching(nr, nc):
                            grid[nr][nc] = grid[r][c]
                            grid[r][c] = background_color
                            changed = True
                            break
    
    # Remove empty rows and columns
    grid = [row for row in grid if any(cell != background_color for cell in row)]
    grid = [list(col) for col in zip(*grid) if any(cell != background_color for cell in col)]
    
    # Ensure all elements touch an edge or another element
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background_color and not is_touching(r, c):
                # Move element to the nearest edge or other element
                nearest_edge = min((0, c), (r, 0), (rows-1, c), (r, cols-1), key=lambda x: abs(x[0]-r) + abs(x[1]-c))
                grid[nearest_edge[0]][nearest_edge[1]] = grid[r][c]
                grid[r][c] = background_color
    
    return grid

def find_largest_region(grid: ColoredGrid) -> Tuple[int, int]:
    largest_color = 0
    largest_size = 0
    for color in range(10):  # 0 to 9
        regions = grid.find_connected_regions(color)
        if regions:
            size = max(len(region) for region in regions)
            if size > largest_size:
                largest_size = size
                largest_color = color
    return largest_color, largest_size

def find_elements_to_preserve(grid: ColoredGrid, background_color: int) -> List[Tuple[int, int, int]]:
    elements = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != background_color and grid.values[r][c] != 0:
                elements.append((grid.values[r][c], r, c))
    return elements

def create_output_grid(elements: List[Tuple[int, int, int]], background_color: int) -> List[List[int]]:
    if not elements:
        return [[background_color]]
    min_row = min(e[1] for e in elements)
    max_row = max(e[1] for e in elements)
    min_col = min(e[2] for e in elements)
    max_col = max(e[2] for e in elements)
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    return [[background_color for _ in range(width)] for _ in range(height)]

def place_preserved_elements(grid: List[List[int]], elements: List[Tuple[int, int, int]]):
    min_row = min(e[1] for e in elements)
    min_col = min(e[2] for e in elements)
    for color, r, c in elements:
        grid[r - min_row][c - min_col] = color

def optimize_grid(grid: List[List[int]]) -> List[List[int]]:
    # Remove empty rows from top and bottom
    while grid and all(cell == grid[0][0] for cell in grid[0]):
        grid.pop(0)
    while grid and all(cell == grid[-1][0] for cell in grid[-1]):
        grid.pop()
    
    # Remove empty columns from left and right
    while grid and all(row[0] == grid[0][0] for row in grid):
        for row in grid:
            row.pop(0)
    while grid and all(row[-1] == grid[0][-1] for row in grid):
        for row in grid:
            row.pop()
    
    return grid
