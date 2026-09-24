from dataclasses import dataclass


@dataclass
class Instruction:
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple = ()

    def __iter__(self):
        return iter(
            [
                self.i_in1,
                self.i_in2,
                self.i_out,
                self.connection_mode,
                self.has_weight,
                self.path_weight,
                self.path_shape,
            ]
        )
