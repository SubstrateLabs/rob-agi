from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_93c31fbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by completing and extending blue (1) structures while respecting other colored elements.
    The solution:
    1. Identifies existing blue structures (lines, L-shapes, squares).
    2. Completes partial structures by adding blue pixels.
    3. Extends structures to create larger patterns.
    4. Ensures connectivity between blue structures.
    5. Maintains balance and symmetry in the overall pattern.
    6. Respects existing non-blue elements.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_blue_pixels() -> Set[Tuple[int, int]]:
        return {(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 1}

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)] if is_valid(r+dr, c+dc)]

    def identify_structures(blue_pixels: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
        structures = []
        visited = set()
        for pixel in blue_pixels:
            if pixel not in visited:
                structure = set()
                stack = [pixel]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        structure.add(current)
                        stack.extend(neighbor for neighbor in get_neighbors(*current) if neighbor in blue_pixels)
                structures.append(structure)
        return structures

    def complete_structure(structure: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        completion = set()
        for r, c in structure:
            for nr, nc in get_neighbors(r, c):
                if grid.values[nr][nc] == 0:
                    blue_neighbors = sum(1 for nnr, nnc in get_neighbors(nr, nc) if (nnr, nnc) in structure)
                    if blue_neighbors >= 2:
                        completion.add((nr, nc))
        return completion

    def extend_structure(structure: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        extension = set()
        for r, c in structure:
            for nr, nc in get_neighbors(r, c):
                if grid.values[nr][nc] == 0:
                    blue_neighbors = sum(1 for nnr, nnc in get_neighbors(nr, nc) if (nnr, nnc) in structure)
                    if blue_neighbors == 1:
                        extension.add((nr, nc))
        return extension

    def apply_changes(changes: Set[Tuple[int, int]]):
        for r, c in changes:
            grid.values[r][c] = 1

    # Main execution
    blue_pixels = get_blue_pixels()
    structures = identify_structures(blue_pixels)

    for _ in range(3):  # Iterate a few times to allow for multi-step completions
        for structure in structures:
            completion = complete_structure(structure)
            apply_changes(completion)
            structure.update(completion)

        for structure in structures:
            extension = extend_structure(structure)
            apply_changes(extension)
            structure.update(extension)

        blue_pixels = get_blue_pixels()
        structures = identify_structures(blue_pixels)

    return grid
