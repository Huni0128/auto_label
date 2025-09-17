from PIL import Image


def get_resample_lanczos():
    """Pillow 10.x 호환: Image.Resampling.LANCZOS / 이전 버전: Image.LANCZOS"""
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


RESAMPLE = get_resample_lanczos()
