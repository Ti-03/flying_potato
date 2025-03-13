import os
import logging
import threading
import queue
import cv2
import numpy as np
from pyzbar.pyzbar import decode

import olympe
from olympe.messages.ardrone3.Piloting import Emergency

DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
FILE_PATH = "DetectedQR.txt"  # File to store any QR data
FRAME_WIDTH, FRAME_HEIGHT = 640, 360  # Display resolution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
log = logging.getLogger(__name__)


def detect_qr_codes(frame):

    decoded_qr_data = None
    for barcode in decode(frame):
        decoded_qr_data = barcode.data.decode("utf-8")
        pts = np.array([barcode.polygon], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 0, 255), 5)
        with open(FILE_PATH, "a") as f:
            f.write(decoded_qr_data + "\n")

    return decoded_qr_data


class ParrotDrone:
    """
    Simple class to handle drone connect/disconnect.
    You can extend this with other drone commands (takeoff, move, etc.).
    """
    def __init__(self, ip_address):
        self.drone = olympe.Drone(ip_address)
        log.info("Connecting to drone at %s...", ip_address)
        if not self.drone.connect():
            log.error("Failed to connect to drone at %s", ip_address)

    def disconnect(self):
        log.info("Disconnecting from drone...")
        self.drone.disconnect()

    def emergency_cut(self):
        """
        Immediately cut the motors (use with caution).
        """
        log.critical("Performing emergency stop!")
        self.drone(Emergency()).wait()


class OlympeStreaming(threading.Thread):
    """
    A thread that continuously captures frames from the drone’s camera via Olympe
    and allows for real-time processing (QR detection, display, etc.).
    """
    def __init__(self, drone):
        super().__init__()
        self.drone = drone
        self.frame_queue = queue.Queue()
        self.flush_queue_lock = threading.Lock()
        self.should_stop = False

        drone.streaming.set_callbacks(
            raw_cb=self.yuv_frame_cb,
            start_cb=self.start_cb,
            end_cb=self.end_cb,
            flush_raw_cb=self.flush_cb,
        )

        log.info("Starting drone video streaming...")
        drone.streaming.start()

    def yuv_frame_cb(self, yuv_frame):
        """
        Callback when a new YUV frame is received.
        We add it to a queue to be processed in `run()`.
        """
        yuv_frame.ref()
        self.frame_queue.put_nowait(yuv_frame)

    def flush_cb(self, stream):
        """
        Called when Olympe restarts the stream.
        Clear any old frames from the queue.
        """
        with self.flush_queue_lock:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait().unref()
        return True

    def start_cb(self):
        log.info("Drone streaming started")

    def end_cb(self):
        log.info("Drone streaming ended")

    def stop(self):
        """
        Signal the thread to exit and stop the drone streaming.
        """
        self.should_stop = True
        self.drone.streaming.stop()
        log.info("Stopping drone video streaming...")

    def run(self):
        """
        Thread loop: constantly pop frames from the queue, decode them to BGR,
        detect QR codes, and display them.
        """
        while not self.should_stop:
            try:
                with self.flush_queue_lock:
                    yuv_frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            cv2_cvt_color_flag = {
                olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
                olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
            }.get(yuv_frame.format(), None)

            if cv2_cvt_color_flag is None:
                yuv_frame.unref()
                continue

            cv2_frame = cv2.cvtColor(
                yuv_frame.as_ndarray(), cv2_cvt_color_flag
            )
            cv2_frame = cv2.resize(cv2_frame, (FRAME_WIDTH, FRAME_HEIGHT))

            qr_data = detect_qr_codes(cv2_frame)
            if qr_data:
                log.info("QR Code Detected: %s", qr_data)

            cv2.imshow("Drone Stream (QR Detection)", cv2_frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.should_stop = True

            yuv_frame.unref()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    log.info("=== Drone QR Detection Script ===")

    my_drone = ParrotDrone(DRONE_IP)

    streamer = OlympeStreaming(my_drone.drone)
    streamer.start()

    try:
        while not streamer.should_stop:
            pass
    except KeyboardInterrupt:
        log.info("Interrupted by user...")
    finally:

        streamer.stop()
        streamer.join()  # wait for the streaming thread to exit
        my_drone.disconnect()

    log.info("=== Script finished ===")
