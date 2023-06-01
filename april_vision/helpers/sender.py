import base64
from threading import Thread
from typing import Callable, List, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from ..marker import Marker
from ..utils import Frame
from ..vision import Processor


class Base64Sender:
    """
    A helper class to encode and send frame data using the publish callback.

    The image data is a base64 encoded JPEG bytestring.
    :param annotated: Controls whether images are annotated with detections
                      before sending. This will copy the frame prior to annotation.
    :param threaded: Controls whether encoding and sending the image is processed
                     in a thread or blocks.
    """
    def __init__(
        self,
        publish_callback: Callable[[str, bytes], None],
        *,
        annotated: bool = True,
        threaded: bool = True,
        aruco_orientation: bool = True,
    ):
        self._publish_callback = publish_callback
        self.use_threads = threaded
        self.send_annotated = annotated
        self._processor = Processor(aruco_orientation=aruco_orientation)

    def annotated_frame_hook(self, frame: Frame, markers: List[Marker]) -> None:
        """
        The hook function called in the detection hook.

        Optionally annotates a copy of the frame, possibly in a background
        thread, depending on the instance settings.
        """
        copied_frame = Frame(
            np.array(frame.grey_frame, copy=True),
            np.array(frame.colour_frame, copy=True),
        )
        if self.send_annotated:
            output_frame = self._processor._annotate(copied_frame, markers)
        else:
            output_frame = copied_frame

        if self.use_threads:
            thread = Thread(
                name="Image send",
                target=self.encode_and_send,
                args=(output_frame.colour_frame,),
                daemon=True,
            )
            thread.start()
        else:
            self.encode_and_send(output_frame.colour_frame)

    def encode_and_send(self, frame: NDArray[np.uint8]) -> None:
        """
        Handle converting a frame to a base64 bytestring and sending it using
        the publish callback.

        Can be run as a thread target.
        """
        if self._publish_callback is None:
            return

        encoded_frame = self.base64_encode_frame(frame)
        if encoded_frame is None:
            return

        encoded_frame = b'data:image/jpeg;base64, ' + encoded_frame

        # The message is sent in a background thread
        try:
            self._publish_callback(
                'camera/annotated' if self.send_annotated else 'camera/image',
                encoded_frame,
            )
        except ValueError:
            pass

    def base64_encode_frame(self, frame: NDArray[np.uint8]) -> Optional[bytes]:
        """Convert image frame to base64 bytestring."""
        ret, image = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        image_bytes = image.tobytes()
        return base64.b64encode(image_bytes)
