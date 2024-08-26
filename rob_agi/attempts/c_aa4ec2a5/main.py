from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_aa4ec2a5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Finds all connected blue (1) regions.
    2. Large blue regions are filled with sky blue (8), with protrusions filled with magenta (6).
    3. Small blue regions remain blue (1).
    4. All blue regions are outlined with red (2).
    5. Creates a grid-wide segmentation with red (2) lines.
    6. Preserves the yellow (4) background where appropriate.
    """
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

    def fill_region(new_grid: ColoredGrid, region: List[Tuple[int, int]], main_color: int, protrusion_color: int):
        main_body = set(region)
        for y, x in region:
            neighbors = [(y+dy, x+dx) for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                         if 0 <= y+dy < new_grid.num_rows and 0 <= x+dx < new_grid.num_cols]
            if len([n for n in neighbors if n in main_body]) <= 2:
                new_grid.values[y][x] = protrusion_color
            else:
                new_grid.values[y][x] = main_color

    def outline_region(new_grid: ColoredGrid, region: List[Tuple[int, int]], outline_color: int):
        for y, x in region:
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < new_grid.num_rows and 0 <= nx < new_grid.num_cols and (ny, nx) not in region:
                    new_grid.values[ny][nx] = outline_color

    def create_segmentation(new_grid: ColoredGrid, segment_size: int):
        for y in range(0, new_grid.num_rows, segment_size):
            for x in range(new_grid.num_cols):
                if new_grid.values[y][x] == 4:
                    new_grid.values[y][x] = 2
        for x in range(0, new_grid.num_cols, segment_size):
            for y in range(new_grid.num_rows):
                if new_grid.values[y][x] == 4:
                    new_grid.values[y][x] = 2

    # Create a new grid filled with yellow (4)
    new_grid = ColoredGrid(values=[[4 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    
    # Find all blue regions
    blue_regions = find_blue_regions(input_grid)
    
    # Process each blue region
    for region in blue_regions:
        if len(region) > 4:  # Adjust this threshold as needed
            fill_region(new_grid, region, 8, 6)  # 8 for main body, 6 for protrusions
        else:
            fill_region(new_grid, region, 1, 1)  # Keep small regions blue
        outline_region(new_grid, region, 2)
    
    # Create grid-wide segmentation
    create_segmentation(new_grid, 5)  # Adjust segment size as needed
    
    # Final pass to ensure yellow background is preserved where needed
    for y in range(input_grid.num_rows):
        for x in range(input_grid.num_cols):
            if input_grid.values[y][x] == 4 and new_grid.values[y][x] == 4:
                new_grid.values[y][x] = 4
    
    return new_grid
