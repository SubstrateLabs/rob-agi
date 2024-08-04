import base64
import copy
import json
from collections import deque
from typing import Annotated, List, Tuple, Set, Dict, Optional, Literal, Callable
from pydantic import BaseModel, Field, field_validator


class ColoredGrid(BaseModel):
    """
    A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive).
    The smallest possible grid size is 1x1 and the largest is 30x30.
    Each integer represents a color.
    0: black, 1: blue, 2: red, 3: green, 4: yellow, 5: gray, 6: magenta, 7: orange, 8: sky, 9: brown
    """

    values: Annotated[
        List[List[Annotated[int, Field(ge=0, le=9)]]], Field(min_length=1, max_length=30, min_items=1, max_items=30)
    ]

    @field_validator("values")
    @classmethod
    def validate_grid(cls, v):
        if not 1 <= len(v) <= 30:
            raise ValueError("Grid must have between 1 and 30 rows")
        if not all(len(row) == len(v[0]) for row in v):
            raise ValueError("All rows must have the same length")
        if not all(0 <= val <= 9 for row in v for val in row):
            raise ValueError("All values must be between 0 and 9")
        return v

    @property
    def is_valid(self) -> bool:
        try:
            self.model_validate(self.model_dump())
            return True
        except ValueError:
            return False

    def __str__(self):
        as_str = "[\n"
        for row in self.values:
            as_str += f" {row},\n"
        as_str += "]"
        return as_str

    @classmethod
    @property
    def colors(cls) -> dict[int, str]:
        return {
            0: "black",
            1: "blue",
            2: "red",
            3: "green",
            4: "yellow",
            5: "gray",
            6: "magenta",
            7: "orange",
            8: "sky",
            9: "brown",
        }

    @classmethod
    @property
    def color_mapping_str(cls) -> str:
        return "\n".join([f"{k}: {v}" for k, v in cls.colors.items()])

    @classmethod
    def value_to_color(cls, val: int) -> str:
        return cls.colors[val]

    def __eq__(self, other):
        return self.values == other.values

    def validate_report(self, expected: "ColoredGrid") -> str:
        result = ""
        result += f"Output:\n"
        result += str(self)
        result += f"\nExpected:\n"
        result += str(expected)
        result += f"\nMatch: {self == expected}\n"
        return result

    ##############################
    # Grid Operations:
    ##############################

    def get_dimensions(self) -> Tuple[int, int]:
        """Return the dimensions of the grid (rows, columns)."""
        return len(self.values), len(self.values[0])

    def get_cell(self, row: int, col: int) -> int:
        """Get the value of a specific cell."""
        return self.values[row][col]

    def set_cell(self, row: int, col: int, value: int) -> None:
        """Set the value of a specific cell."""
        if 0 <= value <= 9:
            self.values[row][col] = value
        else:
            raise ValueError("Cell value must be between 0 and 9")

    def deep_copy(self) -> "ColoredGrid":
        """Return a deep copy of the grid."""
        return copy.deepcopy(self)

    def rotate_90(self, clockwise: bool = True) -> "ColoredGrid":
        """Rotate the grid 90 degrees clockwise or counterclockwise."""
        if clockwise:
            new_values = [list(row) for row in zip(*self.values[::-1])]
        else:
            new_values = [list(row) for row in zip(*self.values)][::-1]
        return ColoredGrid(values=new_values)

    def flip_horizontal(self) -> "ColoredGrid":
        """Flip the grid horizontally."""
        return ColoredGrid(values=[row[::-1] for row in self.values])

    def flip_vertical(self) -> "ColoredGrid":
        """Flip the grid vertically."""
        return ColoredGrid(values=self.values[::-1])

    def count_color(self, color: int) -> int:
        """Count occurrences of a specific color in the grid."""
        return sum(row.count(color) for row in self.values)

    def get_unique_colors(self) -> Set[int]:
        """Return a set of unique colors in the grid."""
        return set(color for row in self.values for color in row)

    def get_color_frequencies(self) -> Dict[int, int]:
        """Return a dictionary of color frequencies."""
        frequencies = {}
        for row in self.values:
            for color in row:
                frequencies[color] = frequencies.get(color, 0) + 1
        return frequencies

    def get_bounding_box(self, color: int) -> Optional[Tuple[int, int, int, int]]:
        """Get the bounding box (top, left, bottom, right) of a specific color."""
        rows, cols = self.get_dimensions()
        color_cells = [(r, c) for r in range(rows) for c in range(cols) if self.values[r][c] == color]
        if not color_cells:
            return None
        top = min(r for r, _ in color_cells)
        bottom = max(r for r, _ in color_cells)
        left = min(c for _, c in color_cells)
        right = max(c for _, c in color_cells)
        return top, left, bottom, right

    def get_symmetry_axes(self) -> Tuple[bool, bool]:
        """Check for horizontal and vertical symmetry in the grid."""
        rows, cols = self.get_dimensions()
        horizontal = all(self.values[r] == self.values[-r - 1] for r in range(rows // 2))
        vertical = all(self.values[r][c] == self.values[r][-c - 1] for r in range(rows) for c in range(cols // 2))
        return horizontal, vertical

    def crop(self, top: int, left: int, bottom: int, right: int) -> "ColoredGrid":
        """Crop the grid to the specified rectangle."""
        return ColoredGrid(values=[row[left : right + 1] for row in self.values[top : bottom + 1]])

    def expand(self, top: int, right: int, bottom: int, left: int, fill_color: int = 0) -> "ColoredGrid":
        """Expand the grid in all directions with a fill color."""
        rows, cols = self.get_dimensions()
        new_rows = rows + top + bottom
        new_cols = cols + left + right
        new_values = [[fill_color] * new_cols for _ in range(new_rows)]
        for r in range(rows):
            for c in range(cols):
                new_values[r + top][c + left] = self.values[r][c]
        return ColoredGrid(values=new_values)

    def replace_color(self, old_color: int, new_color: int) -> "ColoredGrid":
        """Replace all occurrences of one color with another."""
        return ColoredGrid(values=[[new_color if cell == old_color else cell for cell in row] for row in self.values])

    def apply_mask(self, mask: "ColoredGrid", replace_color: int) -> "ColoredGrid":
        """Apply a boolean mask to the grid, replacing masked areas with a color."""
        if self.get_dimensions() != mask.get_dimensions():
            raise ValueError("Mask dimensions must match grid dimensions")
        new_values = [
            [self.values[r][c] if mask.values[r][c] else replace_color for c in range(len(self.values[0]))]
            for r in range(len(self.values))
        ]
        return ColoredGrid(values=new_values)

    def find_pattern(self, pattern: "ColoredGrid") -> List[Tuple[int, int]]:
        """Find all occurrences of a pattern in the grid."""
        rows, cols = self.get_dimensions()
        pattern_rows, pattern_cols = pattern.get_dimensions()
        occurrences = []
        for r in range(rows - pattern_rows + 1):
            for c in range(cols - pattern_cols + 1):
                if all(
                    self.values[r + i][c + j] == pattern.values[i][j]
                    for i in range(pattern_rows)
                    for j in range(pattern_cols)
                ):
                    occurrences.append((r, c))
        return occurrences

    def extract_subgrid(self, top: int, left: int, height: int, width: int) -> "ColoredGrid":
        """Extract a subgrid from the current grid."""
        return ColoredGrid(values=[row[left : left + width] for row in self.values[top : top + height]])

    def tile_grid(self, tiles: int) -> "ColoredGrid":
        """Create a new grid by tiling the current grid."""
        return ColoredGrid(values=[row * tiles for row in self.values] * tiles)

    def add(self, other: "ColoredGrid", modulo: int = 10) -> "ColoredGrid":
        """Add two grids element-wise, with optional modulo."""
        if self.get_dimensions() != other.get_dimensions():
            raise ValueError("Grids must have the same dimensions")
        rows, cols = self.get_dimensions()
        new_values = [[(self.values[r][c] + other.values[r][c]) % modulo for c in range(cols)] for r in range(rows)]
        return ColoredGrid(values=new_values)

    def subtract(self, other: "ColoredGrid", modulo: int = 10) -> "ColoredGrid":
        """Subtract two grids element-wise, with optional modulo."""
        if self.get_dimensions() != other.get_dimensions():
            raise ValueError("Grids must have the same dimensions")
        rows, cols = self.get_dimensions()
        new_values = [[(self.values[r][c] - other.values[r][c]) % modulo for c in range(cols)] for r in range(rows)]
        return ColoredGrid(values=new_values)

    def multiply(self, scalar: int, modulo: int = 10) -> "ColoredGrid":
        """Multiply the grid by a scalar, with optional modulo."""
        return ColoredGrid(values=[[(cell * scalar) % modulo for cell in row] for row in self.values])

    @classmethod
    def from_subgrids(cls, subgrids: List[List["ColoredGrid"]]) -> "ColoredGrid":
        """Create a new grid by composing subgrids."""
        rows = []
        for subgrid_row in subgrids:
            subgrid_heights = [sg.get_dimensions()[0] for sg in subgrid_row]
            if len(set(subgrid_heights)) != 1:
                raise ValueError("All subgrids in a row must have the same height")
            height = subgrid_heights[0]
            for i in range(height):
                rows.append([cell for sg in subgrid_row for cell in sg.values[i]])
        return cls(values=rows)

    def split_into_subgrids(self, rows: int, cols: int) -> List[List["ColoredGrid"]]:
        """Split the grid into a specified number of subgrids."""
        total_rows, total_cols = self.get_dimensions()
        if total_rows % rows != 0 or total_cols % cols != 0:
            raise ValueError("Grid cannot be evenly split into the specified number of subgrids")

        subgrid_height = total_rows // rows
        subgrid_width = total_cols // cols

        return [
            [
                ColoredGrid(
                    values=[
                        self.values[r * subgrid_height + i][c * subgrid_width : (c + 1) * subgrid_width]
                        for i in range(subgrid_height)
                    ]
                )
                for c in range(cols)
            ]
            for r in range(rows)
        ]

    def to_binary(self, threshold: int) -> "ColoredGrid":
        """Convert the grid to binary based on a threshold."""
        return ColoredGrid(values=[[1 if cell >= threshold else 0 for cell in row] for row in self.values])

    def compress_rle(self) -> List[Tuple[int, int]]:
        """Compress the grid using run-length encoding."""
        rle = []
        for row in self.values:
            count = 1
            prev = row[0]
            for cell in row[1:]:
                if cell == prev:
                    count += 1
                else:
                    rle.append((prev, count))
                    prev = cell
                    count = 1
            rle.append((prev, count))
        return rle

    @classmethod
    def from_rle(cls, rle: List[Tuple[int, int]], width: int) -> "ColoredGrid":
        """Create a grid from run-length encoded data."""
        flat = [color for color, count in rle for _ in range(count)]
        rows = [flat[i : i + width] for i in range(0, len(flat), width)]
        return cls(values=rows)

    def get_edge_cells(self) -> List[Tuple[int, int]]:
        """Get all cells on the edge of the grid."""
        rows, cols = self.get_dimensions()
        return (
            [(0, c) for c in range(cols)]
            + [(rows - 1, c) for c in range(cols)]
            + [(r, 0) for r in range(1, rows - 1)]
            + [(r, cols - 1) for r in range(1, rows - 1)]
        )

    def flood_fill(self, row: int, col: int, new_color: int) -> "ColoredGrid":
        """Perform a flood fill starting from a specific cell."""
        rows, cols = self.get_dimensions()
        original_color = self.values[row][col]
        if original_color == new_color:
            return self.deep_copy()

        new_grid = self.deep_copy()
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if new_grid.values[r][c] == original_color:
                new_grid.values[r][c] = new_color
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return new_grid

    def detect_rectangles(self) -> List[Tuple[int, int, int, int, int]]:
        rows, cols = self.get_dimensions()
        rectangles = []
        visited = set()

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited:
                    color = self.values[r][c]
                    right = c
                    bottom = r
                    while right + 1 < cols and self.values[r][right + 1] == color:
                        right += 1

                    while bottom + 1 < rows and all(self.values[bottom + 1][cc] == color for cc in range(c, right + 1)):
                        bottom += 1

                    if (bottom > r or right > c) and color != 0:  # Exclude single cells and color 0
                        rectangles.append((color, r, c, bottom, right))

                    for rr in range(r, bottom + 1):
                        for cc in range(c, right + 1):
                            visited.add((rr, cc))

        return rectangles

    def detect_lines(self) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """Detect lines in the grid, returning (color, list of coordinates)."""
        rows, cols = self.get_dimensions()
        lines = []
        visited = set()

        def dfs(r, c, color):
            stack = [(r, c)]
            line = []
            while stack:
                r, c = stack.pop()
                if (r, c) not in visited and self.values[r][c] == color:
                    visited.add((r, c))
                    line.append((r, c))
                    neighbors = 0
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and self.values[nr][nc] == color:
                            neighbors += 1
                            if (nr, nc) not in visited:
                                stack.append((nr, nc))
                    if neighbors > 2:
                        return None  # Not a line
            return line

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited:
                    color = self.values[r][c]
                    line = dfs(r, c, color)
                    if line and len(line) > 1:
                        lines.append((color, line))

        return lines

    def find_connected_regions(self, color: int) -> List[List[Tuple[int, int]]]:
        rows, cols = self.get_dimensions()
        visited: Set[Tuple[int, int]] = set()
        regions: List[List[Tuple[int, int]]] = []

        def is_valid(r: int, c: int) -> bool:
            return 0 <= r < rows and 0 <= c < cols and self.values[r][c] == color

        def bfs(start_r: int, start_c: int) -> List[Tuple[int, int]]:
            queue = deque([(start_r, start_c)])
            region = []
            while queue:
                r, c = queue.popleft()
                if (r, c) not in visited:
                    visited.add((r, c))
                    region.append((r, c))
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # Right, Left, Down, Up
                        nr, nc = r + dr, c + dc
                        if is_valid(nr, nc) and (nr, nc) not in visited:
                            queue.append((nr, nc))
            return region

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and self.values[r][c] == color:
                    new_region = bfs(r, c)
                    if new_region:
                        regions.append(new_region)

        return regions

    def find_largest_object(self, color: int) -> List[Tuple[int, int]]:
        """Find the largest contiguous object of a given color."""
        regions = self.find_connected_regions(color)
        return max(regions, key=len) if regions else []

    def diff(self, other: "ColoredGrid") -> "ColoredGrid":
        """Return a new grid highlighting the differences between two grids."""
        if self.get_dimensions() != other.get_dimensions():
            raise ValueError("Grids must have the same dimensions")
        rows, cols = self.get_dimensions()
        diff_values = [[1 if self.values[r][c] != other.values[r][c] else 0 for c in range(cols)] for r in range(rows)]
        return ColoredGrid(values=diff_values)

    def similarity_score(self, other: "ColoredGrid") -> float:
        """Calculate a similarity score between two grids."""
        if self.get_dimensions() != other.get_dimensions():
            raise ValueError("Grids must have the same dimensions")
        rows, cols = self.get_dimensions()
        total_cells = rows * cols
        matching_cells = sum(self.values[r][c] == other.values[r][c] for r in range(rows) for c in range(cols))
        return matching_cells / total_cells

    def scale(self, factor: int) -> "ColoredGrid":
        """Scale the grid by a given factor."""
        return ColoredGrid(
            values=[[cell for cell in row for _ in range(factor)] for row in self.values for _ in range(factor)]
        )

    def find_repeating_pattern(self) -> Optional["ColoredGrid"]:
        """Attempt to find a repeating sub pattern in the grid."""
        rows, cols = self.get_dimensions()
        for pattern_height in range(1, rows + 1):
            for pattern_width in range(1, cols + 1):
                if rows % pattern_height == 0 and cols % pattern_width == 0:
                    pattern = self.extract_subgrid(0, 0, pattern_height, pattern_width)
                    if all(
                        self.values[r][c] == pattern.values[r % pattern_height][c % pattern_width]
                        for r in range(rows)
                        for c in range(cols)
                    ):
                        return pattern
        return None

    def extrapolate_sequence(
        self,
        direction: Literal["right", "left", "up", "down", "up-right", "up-left", "down-right", "down-left"],
        steps: int = 1,
    ) -> "ColoredGrid":
        """Attempt to extrapolate the grid in a given direction based on patterns."""
        rows, cols = self.get_dimensions()
        new_values = [row[:] for row in self.values]

        for _ in range(steps):
            if direction in ["right", "left"]:
                step = 1 if direction == "right" else -1
                for r in range(rows):
                    new_values[r].append(new_values[r][-1] if step == 1 else new_values[r][0])
            elif direction in ["down", "up"]:
                step = 1 if direction == "down" else -1
                new_values.append(new_values[-1][:] if step == 1 else new_values[0][:])
            else:
                # Diagonal directions
                dr = 1 if "down" in direction else -1
                dc = 1 if "right" in direction else -1
                new_values.append([0] * (cols + 1))
                for r in range(rows):
                    new_values[r].append(0)
                new_values[-1][-1] = new_values[-1 - dr][-1 - dc]

            rows, cols = len(new_values), len(new_values[0])
        return ColoredGrid(values=new_values)

    def find_center_of_mass(self) -> Tuple[float, float]:
        rows, cols = self.get_dimensions()
        total_mass = sum(sum(row) for row in self.values)
        if total_mass == 0:
            return rows / 2 - 0.5, cols / 2 - 0.5

        row_sum = sum(r * sum(row) for r, row in enumerate(self.values))
        col_sum = sum(c * self.values[r][c] for r in range(rows) for c in range(cols))

        return row_sum / total_mass, col_sum / total_mass

    def to_base64(self) -> str:
        """Convert the grid to a base64 string representation."""
        return base64.b64encode(json.dumps(self.values).encode()).decode()

    @classmethod
    def from_base64(cls, base64_str: str) -> "ColoredGrid":
        """Create a grid from a base64 string representation."""
        values = json.loads(base64.b64decode(base64_str).decode())
        return cls(values=values)

    def find_color_transitions(self) -> List[Tuple[int, int, int, int]]:
        """Find all color transitions in the grid."""
        rows, cols = self.get_dimensions()
        transitions = []
        for r in range(rows):
            for c in range(cols):
                current = self.values[r][c]
                if c < cols - 1 and self.values[r][c + 1] != current:
                    transitions.append((r, c, current, self.values[r][c + 1]))
                if r < rows - 1 and self.values[r + 1][c] != current:
                    transitions.append((r, c, current, self.values[r + 1][c]))
        return transitions

    def apply_cellular_automaton(self, rule: Callable[[List[int]], int]) -> "ColoredGrid":
        rows, cols = self.get_dimensions()
        new_values = [[0 for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                neighbors = [
                    self.values[nr][nc]
                    for nr in range(r - 1, r + 2)
                    for nc in range(c - 1, c + 2)
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) != (r, c)
                ]
                new_values[r][c] = rule(neighbors)

        return ColoredGrid(values=new_values)

    def apply_function_to_regions(self, func: Callable[[List[Tuple[int, int]]], int]) -> "ColoredGrid":
        new_grid = self.deep_copy()
        visited = set()
        rows, cols = self.get_dimensions()

        def get_region(r, c, color):
            region = []
            stack = [(r, c)]
            while stack:
                r, c = stack.pop()
                if (r, c) not in visited and new_grid.values[r][c] == color:
                    visited.add((r, c))
                    region.append((r, c))
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Only orthogonal
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            stack.append((nr, nc))
            return region

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited:
                    color = new_grid.values[r][c]
                    region = get_region(r, c, color)
                    new_color = func(region)
                    for rr, cc in region:
                        new_grid.values[rr][cc] = new_color

        return new_grid
