from typing import Annotated, List
from pydantic import BaseModel, Field, field_validator


class ColoredGrid(BaseModel):
    """
    A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive).
    The smallest possible grid size is 1x1 and the largest is 30x30.
    Each integer represents a color.
    0: black, 1: blue, 2: red, 3: green, 4: yellow, 5: gray, 6: magenta, 7: orange, 8: sky, 9: brown
    """

    values: Annotated[List[List[int]], Field(min_length=1, max_length=30)]

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
