from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_aa4ec2a5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Initializes the output grid with yellow (4) and red (2) edges.
    2. Identifies and classifies blue (1) regions based on size and shape.
    3. Processes large regions: outlines with red (2), fills with sky blue (8), and marks protrusions with magenta (6).
    4. Processes small regions: outlines with red (2) and keeps interior blue (1).
    5. Creates a segmentation structure with red (2) lines.
    6. Applies color transition rules and makes context-sensitive adjustments.
    7. Preserves the yellow (4) background where appropriate.
    """
    def initialize_grid(input_grid: ColoredGrid) -> ColoredGrid:
        new_grid = ColoredGrid(values=[[4 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
        for x in range(input_grid.num_cols):
            new_grid.values[0][x] = 2
        for y in range(input_grid.num_rows):
            new_grid.values[y][0] = 2
        return new_grid

    def find_blue_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
        visited = set()
        regions = []
        for y in range(grid.num_rows):
            for x in range(grid.num_cols):
                if grid.values[y][x] == 1 and (y, x) not in visited:
                    region = []
                    stack = [(y, x)]
                    while stack:
                        cy, cx = stack.pop()
                        if (cy, cx) not in visited and grid.values[cy][cx] == 1:
                            visited.add((cy, cx))
                            region.append((cy, cx))
                            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < grid.num_rows and 0 <= nx < grid.num_cols:
                                    stack.append((ny, nx))
                    regions.append(region)
        return regions

    def process_large_region(new_grid: ColoredGrid, region: List[Tuple[int, int]]):
        for y, x in region:
            new_grid.values[y][x] = 8  # Fill with sky blue
        
        # Mark protrusions
        for y, x in region:
            neighbors = [(y+dy, x+dx) for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                         if 0 <= y+dy < new_grid.num_rows and 0 <= x+dx < new_grid.num_cols and (y+dy, x+dx) in region]
            if len(neighbors) <= 2:
                new_grid.values[y][x] = 6  # Mark as magenta

        # Outline the region
        outline_region(new_grid, region, 2)

    def process_small_region(new_grid: ColoredGrid, region: List[Tuple[int, int]]):
        if len(region) == 1:
            y, x = region[0]
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < new_grid.num_rows and 0 <= nx < new_grid.num_cols:
                        new_grid.values[ny][nx] = 2 if (dy, dx) != (0, 0) else 1
        else:
            for y, x in region:
                new_grid.values[y][x] = 1
            outline_region(new_grid, region, 2)

    def outline_region(new_grid: ColoredGrid, region: List[Tuple[int, int]], outline_color: int):
        for y, x in region:
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < new_grid.num_rows and 0 <= nx < new_grid.num_cols and (ny, nx) not in region:
                    new_grid.values[ny][nx] = outline_color

    def create_segmentation(new_grid: ColoredGrid):
        for y in range(5, new_grid.num_rows, 5):
            for x in range(new_grid.num_cols):
                if new_grid.values[y][x] == 4:
                    new_grid.values[y][x] = 2
        for x in range(5, new_grid.num_cols, 5):
            for y in range(new_grid.num_rows):
                if new_grid.values[y][x] == 4:
                    new_grid.values[y][x] = 2

    def apply_color_transitions(new_grid: ColoredGrid):
        color_priority = {2: 0, 8: 1, 6: 2, 1: 3, 4: 4}
        for y in range(new_grid.num_rows):
            for x in range(new_grid.num_cols):
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < new_grid.num_rows and 0 <= nx < new_grid.num_cols:
                        if color_priority[new_grid.values[y][x]] < color_priority[new_grid.values[ny][nx]]:
                            new_grid.values[ny][nx] = new_grid.values[y][x]

    # Initialize the new grid
    new_grid = initialize_grid(input_grid)
    
    # Find all blue regions
    blue_regions = find_blue_regions(input_grid)
    
    # Process each blue region
    for region in blue_regions:
        if len(region) > 4:
            process_large_region(new_grid, region)
        else:
            process_small_region(new_grid, region)
    
    # Create segmentation structure
    create_segmentation(new_grid)
    
    # Apply color transitions
    apply_color_transitions(new_grid)
    
    # Final pass to ensure yellow background is preserved where needed
    for y in range(input_grid.num_rows):
        for x in range(input_grid.num_cols):
            if input_grid.values[y][x] == 4 and new_grid.values[y][x] == 4:
                new_grid.values[y][x] = 4
    
    return new_grid
