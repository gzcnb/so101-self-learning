"""Share fresh camera frames between the Orange Pi controller and the local console."""

import fcntl
from pathlib import Path
import threading
import time

import cv2

from hover_probe import CAMERA, CHECKS, WRIST


def camera_loop(name, device, directory, stop):
    number = int(Path(device).resolve().name.removeprefix('video'))
    stream = cv2.VideoCapture(number, cv2.CAP_V4L2)
    try:
        stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while not stop.is_set():
            ok, frame = stream.read()
            if not ok:
                stop.wait(0.2)
                continue
            encoded, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
            if encoded:
                scratch = directory / f'{name}.part'
                scratch.write_bytes(jpeg.tobytes())
                scratch.replace(directory / f'{name}.jpg')
            stop.wait(0.08)
    finally:
        stream.release()


def main():
    directory = CHECKS / 'swap_console_frames'
    directory.mkdir(mode=0o700, exist_ok=True)
    with (directory / 'camera.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        stop = threading.Event()
        threads = [threading.Thread(target=camera_loop, args=(name, device, directory, stop), daemon=True)
                   for name, device in (('fixed', CAMERA), ('wrist', WRIST))]
        for thread in threads:
            thread.start()
        try:
            while any(thread.is_alive() for thread in threads):
                heartbeat = directory / 'heartbeat'
                if not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 40:
                    break
                time.sleep(1)
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=2)


if __name__ == '__main__':
    main()
