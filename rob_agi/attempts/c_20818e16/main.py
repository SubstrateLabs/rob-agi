from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_20818e16(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by identifying significant colored regions,
    preserving their relative positions, and minimizing the output grid size.
    
    1. Identify significant regions using flood fill.
    2. Determine bounding boxes for each region.
    3. Calculate new grid dimensions to encompass all regions.
    4. Transfer regions to the new grid, maintaining relative positions.
    5. Optimize grid size by trimming empty rows and columns.
    
    Returns a new ColoredGrid with the transformed grid.
    """
    regions = identify_regions(input_grid)
    bounding_boxes = get_bounding_boxes(regions)
    new_dimensions = calculate_new_dimensions(bounding_boxes)
    new_grid = create_new_grid(regions, bounding_boxes, new_dimensions)
    optimized_grid = trim_grid(new_grid)
    return ColoredGrid(values=optimized_grid)

def identify_regions(grid: ColoredGrid) -> List[Tuple[int, Set[Tuple[int, int]]]]:
    regions = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    def flood_fill(r: int, c: int, color: int) -> Set[Tuple[int, int]]:
        region = set()
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and 0 <= r < rows and 0 <= c < cols and grid.values[r][c] == color:
                visited.add((r, c))
                region.add((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((r + dr, c + dc))
        return region

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                region = flood_fill(r, c, grid.values[r][c])
                regions.append((grid.values[r][c], region))
    
    return regions

def get_bounding_boxes(regions: List[Tuple[int, Set[Tuple[int, int]]]]) -> List[Tuple[int, int, int, int, int]]:
    return [
        (
            color,
            min(r for r, _ in region),
            min(c for _, c in region),
            max(r for r, _ in region),
            max(c for _, c in region)
        )
        for color, region in regions
    ]

def calculate_new_dimensions(bounding_boxes: List[Tuple[int, int, int, int, int]]) -> Tuple[int, int]:
    if not bounding_boxes:
        return 0, 0
    min_r = min(box[1] for box in bounding_boxes)
    min_c = min(box[2] for box in bounding_boxes)
    max_r = max(box[3] for box in bounding_boxes)
    max_c = max(box[4] for box in bounding_boxes)
    return max_r - min_r + 1, max_c - min_c + 1

def create_new_grid(regions: List[Tuple[int, Set[Tuple[int, int]]]], 
                    bounding_boxes: List[Tuple[int, int, int, int, int]], 
                    dimensions: Tuple[int, int]) -> List[List[int]]:
    rows, cols = dimensions
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    min_r = min(box[1] for box in bounding_boxes)
    min_c = min(box[2] for box in bounding_boxes)
    
    for color, region in regions:
        for r, c in region:
            new_r, new_c = r - min_r, c - min_c
            new_grid[new_r][new_c] = color
    
    return new_grid

def trim_grid(grid: List[List[int]]) -> List[List[int]]:
    rows = [r for r, row in enumerate(grid) if any(cell != 0 for cell in row)]
    if not rows:
        return [[]]
    cols = [c for c in range(len(grid[0])) if any(row[c] != 0 for row in grid)]
    return [[grid[r][c] for c in cols] for r in rows]
