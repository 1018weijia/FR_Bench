import cv2
import numpy as np
import os
import subprocess
import pickle
import pdb
import shutil


def _resolve_ffmpeg_executable() -> str:
    """Resolve ffmpeg executable from env/path, with imageio_ffmpeg fallback."""
    ffmpeg_from_env = os.environ.get("FFMPEG_BINARY")
    if ffmpeg_from_env:
        return ffmpeg_from_env

    ffmpeg_in_path = shutil.which("ffmpeg")
    if ffmpeg_in_path:
        return ffmpeg_in_path

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise FileNotFoundError(
            "ffmpeg executable not found. Install ffmpeg or imageio-ffmpeg, "
            "or set FFMPEG_BINARY to a valid ffmpeg path."
        ) from e


def images_to_video(imgs: np.ndarray, out_path: str, fps: float = 30.0, is_rgb: bool = True) -> None:
    if (not isinstance(imgs, np.ndarray) or imgs.ndim != 4 or imgs.shape[3] not in (3, 4)):
        raise ValueError("imgs must be a numpy.ndarray of shape (N, H, W, C), with C equal to 3 or 4.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_frames, H, W, C = imgs.shape
    if C == 3:
        pixel_format = "rgb24" if is_rgb else "bgr24"
    else:
        pixel_format = "rgba"
    ffmpeg_exe = _resolve_ffmpeg_executable()
    command = [
        ffmpeg_exe,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        pixel_format,
        "-video_size",
        f"{W}x{H}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        "-crf",
        "23",
    ]
    ffmpeg_threads = os.environ.get("FFMPEG_THREADS")
    if ffmpeg_threads:
        command.extend(["-threads", str(int(ffmpeg_threads))])
    command.append(f"{out_path}")
    ffmpeg = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
    )
    ffmpeg.stdin.write(imgs.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise IOError(f"Cannot open ffmpeg. Please check the output path and ensure ffmpeg is supported.")

    if os.environ.get("ROBOTWIN_VERBOSE_VIDEO_SAVE") == "1":
        print(
            f"🎬 Video is saved to `{out_path}`, containing \033[94m{n_frames}\033[0m frames at {W}×{H} resolution and {fps} FPS."
        )
