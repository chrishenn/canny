from torch import Tensor

from canny_chenn import Canny


def test_smoke() -> None:
    data = Tensor(
        [
            [
                [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
            ]
        ]
    )
    assert isinstance(Canny()(data), Tensor), "Smoke test failed on simple tensor input"
