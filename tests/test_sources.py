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


from src.pipeline.sources import VideoFileSource


def test_video_file_source_reads_frames(sample_mp4):
    src = VideoFileSource(str(sample_mp4))
    assert src.is_seekable is True
    assert src.frame_count >= 9          # 인코더가 마지막 1프레임 누락할 수 있음
    n = 0
    while True:
        f = src.read()
        if f is None:
            break
        assert f.shape[2] == 3
        n += 1
    assert n >= 9
    src.release()


def test_video_file_source_seek(sample_mp4):
    src = VideoFileSource(str(sample_mp4))
    src.seek(5)
    f = src.read()
    assert f is not None
    src.release()
