
@dataclass
class Outer:

    @staticmethod
    def calculate(x: int) -> int:
        return x * 2

    class Inner:

        @classmethod
        def process(cls, data: str):

            def nested_helper():
                return data.strip()

            return nested_helper()
