import sys
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
    squared = scaled ** 2
    avgsq = np.average(squared, axis=0)
    return np.sqrt(avgsq).astype('uint8')


# Returns a ndarray cointaining a column for every frame taken from the argument (image sequence).
def collect_frames(_mvcap):
    res = []
    read, frame = _mvcap.read()
    while read:
        res.append(frame_avg(frame))
        for _ in range(30):  # The argument of range() is the number of skipped frames for every iteration.
            # Since i only have to skip, i just grab the frame
            if read:
                read = _mvcap.grab()
        read, frame = _mvcap.read()
    return res


def _task(fr):
    res = frame_avg(fr)
    return res


def movie_iter(movie_name, frames_to_skip):
    movie = cv2.VideoCapture(movie_name)
    read, frame = movie.read()
    while read:
        yield frame
        for i in range(frames_to_skip):
            _ = movie.grab()
        read, frame = movie.read()


def _indexedtask(fc, fr):
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
            executor.submit(_indexedtask, framecount, frame).add_done_callback(_callback)
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


# Returns a ndarray cointaining a column for every frame taken from the argument (image sequence).
def collect_frames_mpmap(_moviename):
    """
    Prettier version of collect_frames_mp. It's a ~0.5 seconds slower and puts way more stress on the CPU
    The difference seems negligible on I7-8750h, but performance may worsen way more with slower CPUs
    :param _moviename:
    :return:
    """

    with ProcessPoolExecutor() as executor:
        resiter = executor.map(_task, movie_iter(_moviename, 30))
        executor.shutdown(True)
    res = []
    for i in resiter:
        res.append(i)
    return res


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("No arg!")
        exit(-1)
    print("START")
    start = datetime.now()
    complete = collect_frames_mpmap(sys.argv[1])
    print("COLLECTED")
    collectionend = datetime.now()
    c = np.array(complete)
    cc = c.swapaxes(0, 1)
    barcodeimg = Image.fromarray(cc, mode='RGB')
    imagegen = datetime.now()
    barcodeimg.save('barcode.jpg')  # Name of your output
    end = datetime.now()
    print(f"Frame collection: {(collectionend - start).total_seconds()}")
    print(
        f"Image generation: {(imagegen - collectionend).total_seconds()} [Total: {(imagegen - start).total_seconds()}]")
    print(f"Total: {(end - start).total_seconds()}")
