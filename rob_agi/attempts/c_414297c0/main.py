from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_414297c0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by selecting the most frequent non-black color as background,
    preserving all other colored elements and formations, and arranging them
    while maintaining their relative positions and relationships.
    
    1. Analyzes the input grid to identify the background color and elements.
    2. Detects connected regions and analyzes element relationships.
    3. Determines the optimal output grid size.
    4. Places elements in the new grid, maintaining relative positions and relationships.
    5. Optimizes element placement and handles special cases.
    6. Performs final adjustments to ensure proper element placement and border.
    
    Returns a new ColoredGrid object with the transformed grid.
    """
    # Step 1: Analyze input and select background color
    background_color = select_background_color(input_grid)
    
    # Step 2: Detect elements and analyze relationships
    elements = detect_elements(input_grid, background_color)
    analyze_relationships(elements)
    
    # Step 3: Determine output grid size
    output_size = determine_output_size(input_grid, elements)
    
    # Step 4: Create initial output grid and place elements
    output_grid = create_initial_grid(output_size, background_color)
    place_elements(output_grid, elements, background_color)
    
    # Step 5: Optimize placement and handle special cases
    optimize_placement(output_grid, elements, background_color)
    
    # Step 6: Perform final adjustments
    final_grid = final_adjustments(output_grid, background_color)
    
    # Create and return the final ColoredGrid object
    return ColoredGrid(values=final_grid)

def select_background_color(grid: ColoredGrid) -> int:
    """Selects the most frequent non-black color as the background."""
    color_counts = grid.get_color_frequencies()
    if 0 in color_counts:
        del color_counts[0]  # Remove black (0) from consideration
    return max(color_counts, key=color_counts.get) if color_counts else 1  # Default to blue if no other colors

def detect_elements(grid: ColoredGrid, background_color: int) -> List[Dict]:
    """Detects connected regions of non-background colors."""
    rows, cols = grid.get_dimensions()
    visited = set()
    elements = []

    def flood_fill(r, c, color):
        stack = [(r, c)]
        region = []
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and 0 <= r < rows and 0 <= c < cols and grid.values[r][c] == color:
                visited.add((r, c))
                region.append((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((r + dr, c + dc))
        return region

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != background_color and grid.values[r][c] != 0:
                region = flood_fill(r, c, grid.values[r][c])
                elements.append({"color": grid.values[r][c], "coords": region})

    return elements

def analyze_relationships(elements: List[Dict]):
    """Analyzes relative positions and relationships between elements."""
    for i, elem1 in enumerate(elements):
        elem1["centroid"] = (sum(r for r, _ in elem1["coords"]) / len(elem1["coords"]),
                             sum(c for _, c in elem1["coords"]) / len(elem1["coords"]))
        elem1["relationships"] = []
        for j, elem2 in enumerate(elements):
            if i != j:
                dx = elem2["centroid"][1] - elem1["centroid"][1]
                dy = elem2["centroid"][0] - elem1["centroid"][0]
                direction = ""
                if abs(dx) > abs(dy):
                    direction = "right" if dx > 0 else "left"
                else:
                    direction = "below" if dy > 0 else "above"
                elem1["relationships"].append({"id": j, "direction": direction})

def determine_output_size(input_grid: ColoredGrid, elements: List[Dict]) -> Tuple[int, int]:
    """Determines the optimal output grid size."""
    input_rows, input_cols = input_grid.get_dimensions()
    min_row = min(coord[0] for elem in elements for coord in elem["coords"])
    max_row = max(coord[0] for elem in elements for coord in elem["coords"])
    min_col = min(coord[1] for elem in elements for coord in elem["coords"])
    max_col = max(coord[1] for elem in elements for coord in elem["coords"])
    
    height = max_row - min_row + 3  # Add 2 for border
    width = max_col - min_col + 3   # Add 2 for border
    
    # Ensure output is not larger than input
    return (min(height, input_rows), min(width, input_cols))

def create_initial_grid(size: Tuple[int, int], background_color: int) -> List[List[int]]:
    """Creates the initial output grid."""
    rows, cols = size
    return [[background_color for _ in range(cols)] for _ in range(rows)]

def place_elements(grid: List[List[int]], elements: List[Dict], background_color: int):
    """Places elements in the output grid, maintaining relative positions."""
    rows, cols = len(grid), len(grid[0])
    for elem in elements:
        min_r = min(r for r, _ in elem["coords"])
        min_c = min(c for _, c in elem["coords"])
        for r, c in elem["coords"]:
            new_r = r - min_r + 1
            new_c = c - min_c + 1
            if 0 < new_r < rows - 1 and 0 < new_c < cols - 1:
                grid[new_r][new_c] = elem["color"]

def optimize_placement(grid: List[List[int]], elements: List[Dict], background_color: int):
    """Optimizes element placement and handles special cases."""
    rows, cols = len(grid), len(grid[0])
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid[r][c] != background_color:
                # Ensure elements are not touching
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    if grid[r+dr][c+dc] != background_color and grid[r+dr][c+dc] != grid[r][c]:
                        # Move the element if it's touching another non-background element
                        for mr, mc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            if grid[r+mr][c+mc] == background_color:
                                grid[r+mr][c+mc], grid[r][c] = grid[r][c], background_color
                                break

def final_adjustments(grid: List[List[int]], background_color: int) -> List[List[int]]:
    """Performs final adjustments to ensure proper element placement and border."""
    rows, cols = len(grid), len(grid[0])
    
    # Ensure border
    for r in range(rows):
        grid[r][0] = grid[r][-1] = background_color
    for c in range(cols):
        grid[0][c] = grid[-1][c] = background_color
    
    # Remove unnecessary background-only rows and columns
    row_keep = [any(cell != background_color for cell in row) for row in grid]
    col_keep = [any(grid[r][c] != background_color for r in range(rows)) for c in range(cols)]
    
    # Ensure border rows and columns are kept
    row_keep[0] = row_keep[-1] = True
    col_keep[0] = col_keep[-1] = True
    
    # Create new grid with only necessary rows and columns
    new_grid = [[grid[r][c] for c in range(cols) if col_keep[c]] for r in range(rows) if row_keep[r]]
    
    return new_grid
