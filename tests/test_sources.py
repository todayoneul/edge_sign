import numpy as np
from src.pipeline.sources import ImageSource


def test_image_source_reads_same_frame(sample_image):
    src = ImageSource(str(sample_image))
    f1 = src.read()
    f2 = src.read()
    assert f1 is not None and f1.shape == (48, 64, 3)
    assert np.array_equal(f1, f2)        # 정지: 항상 동일 프레임
    assert src.is_seekable is False
    src.release()
