import copy
from typing import Annotated, List, Tuple, Set, Dict, Optional, Literal
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

    def rotate_90(self, clockwise: bool = True) -> "ColoredGrid":
        """Rotate the grid 90 degrees clockwise or counterclockwise."""
        if clockwise:
            new_values = list(zip(*self.values[::-1]))
        else:
            new_values = list(zip(*self.values))[::-1]
        return ColoredGrid(values=[list(row) for row in new_values])

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

    def get_bounding_box(self, color: int) -> Tuple[int, int, int, int]:
        """Get the bounding box (top, left, bottom, right) of a specific color."""

    def find_connected_regions(self, color: int) -> List[List[Tuple[int, int]]]:
        """Find all connected regions of a specific color."""

    def get_symmetry_axes(self) -> Tuple[bool, bool]:
        """Check if the grid has horizontal or vertical symmetry."""

    ##############################
    # Grid Transformations
    ##############################
    def crop(self, top: int, left: int, bottom: int, right: int) -> "ColoredGrid":
        """Crop the grid to the specified rectangle."""

    def expand(self, top: int, right: int, bottom: int, left: int, fill_color: int = 0) -> "ColoredGrid":
        """Expand the grid in all directions with a fill color."""

    def replace_color(self, old_color: int, new_color: int) -> "ColoredGrid":
        """Replace all occurrences of one color with another."""

    def apply_mask(self, mask: "ColoredGrid", replace_color: int) -> "ColoredGrid":
        """Apply a boolean mask to the grid, replacing masked areas with a color."""

    ##############################
    # Grid Transformations
    ##############################

    def find_pattern(self, pattern: "ColoredGrid") -> List[Tuple[int, int]]:
        """Find all occurrences of a pattern in the grid."""

    def extract_subgrid(self, top: int, left: int, height: int, width: int) -> "ColoredGrid":
        """Extract a subgrid from the current grid."""

    def tile_grid(self, tiles: int) -> "ColoredGrid":
        """Create a new grid by tiling the current grid."""

    ##############################
    # Mathematical Operations
    ##############################
    def add(self, other: "ColoredGrid", modulo: int = 10) -> "ColoredGrid":
        """Add two grids element-wise, with optional modulo."""

    def subtract(self, other: "ColoredGrid", modulo: int = 10) -> "ColoredGrid":
        """Subtract two grids element-wise, with optional modulo."""

    def multiply(self, scalar: int, modulo: int = 10) -> "ColoredGrid":
        """Multiply the grid by a scalar, with optional modulo."""

    ##############################
    # Grid Composition
    ##############################
    @classmethod
    def from_subgrids(cls, subgrids: List[List["ColoredGrid"]]) -> "ColoredGrid":
        """Create a new grid by composing subgrids."""

    def split_into_subgrids(self, rows: int, cols: int) -> List[List["ColoredGrid"]]:
        """Split the grid into a specified number of subgrids."""

    ##############################
    # Utility Functions:
    ##############################
    def to_binary(self, threshold: int) -> "ColoredGrid":
        """Convert the grid to binary based on a threshold."""

    def compress_rle(self) -> List[Tuple[int, int]]:
        """Compress the grid using run-length encoding."""

    @classmethod
    def from_rle(cls, rle: List[Tuple[int, int]], width: int) -> "ColoredGrid":
        """Create a grid from run-length encoded data."""

    def get_edge_cells(self) -> List[Tuple[int, int]]:
        """Get all cells on the edge of the grid."""

    def flood_fill(self, row: int, col: int, new_color: int) -> "ColoredGrid":
        """Perform a flood fill starting from a specific cell."""

    ##############################
    # Shape and Object Detection
    ##############################
    def detect_rectangles(self) -> List[Tuple[int, int, int, int, int]]:
        """Detect rectangles in the grid, returning (color, top, left, bottom, right)."""

    def detect_lines(self) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """Detect lines in the grid, returning (color, list of coordinates)."""

    def find_largest_object(self, color: int) -> List[Tuple[int, int]]:
        """Find the largest contiguous object of a given color."""

    ##############################
    # Grid Comparison and Difference:
    ##############################
    def diff(self, other: "ColoredGrid") -> "ColoredGrid":
        """Return a new grid highlighting the differences between two grids."""

    def similarity_score(self, other: "ColoredGrid") -> float:
        """Calculate a similarity score between two grids."""

    ##############################
    # Advanced Transformations:
    ##############################
    def scale(self, factor: int) -> "ColoredGrid":
        """Scale the grid by a given factor."""

    def skew(self, x_factor: float, y_factor: float) -> "ColoredGrid":
        """Apply a skew transformation to the grid."""

    ##############################
    # Pattern and Sequence Analysis:
    ##############################
    def find_repeating_pattern(self) -> Optional["ColoredGrid"]:
        """Attempt to find a repeating sub pattern in the grid."""

    def extrapolate_sequence(
        self, direction: Literal["right", "left", "up", "down", "up-right", "up-left", "down-right", "down-left"]
    ) -> "ColoredGrid":
        """
        Attempt to extrapolate the grid in a given direction based on patterns.
        The direction can be "right", "left", "up", "down", "up-right", "up-left", "down-right", "down-left"
        """

    ##############################
    # Geometric Operations:
    ##############################
    def find_center_of_mass(self) -> Tuple[float, float]:
        """Find the center of mass of the grid, treating colors as weights."""
