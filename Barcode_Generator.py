from collections import OrderedDict
from concurrent.futures import Future
from concurrent.futures.process import ProcessPoolExecutor
from datetime import datetime
from typing import Dict

import cv2
import numpy as np
from PIL import Image


# Returns the vectorized mean of the argument (an image).
def frame_avg(img):
    scaled = img.astype('uint32')
    squared = scaled**2
    avgsq = np.average(squared, axis=0)
    return np.sqrt(avgsq).astype('uint8')


# Returns a ndarray cointaining a column for every frame taken from the argument (image sequence).
def collect_frames(_mvcap):
    res = []
    # collo di bottiglia
    read, frame = _mvcap.read()
    while read:
        res.append(frame_avg(frame))
        for _ in range(30):           # The argument of range() is the number of skipped frames for every iteration.
            # Since i only have to skip, i just grab the frame
            if read:
                read = _mvcap.grab()
        read, frame = _mvcap.read()
    return res


def _task(fc, fr):
    res = frame_avg(fr)
    return fc, res


# Returns a ndarray cointaining a column for every frame taken from the argument (image sequence).
def collect_frames_mp(_mvcap):
    """
    Multiprocess version of #collect_frames. It cuts ~10 seconds on i7-8750h delegating processing to a pool of
    processors. Less powerful CPUs could make this run slower than the standard version.
    :param _mvcap:
    :return:
    """
    avgdict: Dict[int, np.ndarray] = OrderedDict({})
    framecount = 0
    read, frame = _mvcap.read()

    def _callback(fut: Future):
        index, value = fut.result()
        avgdict[index] = value

    with ProcessPoolExecutor() as executor:
        while read:
            executor.submit(_task, framecount, frame).add_done_callback(_callback)
            framecount += 1
            for _ in range(30):      # The argument of range() is the number of skipped frames for every iteration.
                if read:
                    read = _mvcap.grab()
            if read:
                read, frame = _mvcap.read()
        executor.shutdown(True)
    res = []
    for i in range(framecount):
        res.append(avgdict[i])
    return res


if __name__ == '__main__':
    print("START")
    start = datetime.now()
    mvcap = cv2.VideoCapture("C:\\Users\\Francesco Calcagnini\\Downloads\\[Akuma+Omnivium]_Nisemonogatari_-_06_[BD][720p][Hi444PP][Opus][5FCC2F4F].mkv")  # VideoCapture take as argument any video files, image sequences or cameras.
    complete = collect_frames_mp(mvcap)
    print("COLLECTED")
    collectionend = datetime.now()
    c = np.array(complete)
    cc = c.swapaxes(0, 1)
    barcodeimg = Image.fromarray(cc, mode='RGB')
    imagegen = datetime.now()
    barcodeimg.save('barcode.jpg')  # Name of your output
    end = datetime.now()
    print(f"Frame collection: {(collectionend - start).total_seconds()}")
    print(f"Image generation: {(imagegen - collectionend).total_seconds()} [Total: {(imagegen - start).total_seconds()}]")
    print(f"Total: {(end - start).total_seconds()}")
