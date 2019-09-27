import sys
from concurrent.futures.process import ProcessPoolExecutor
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from multiprocessing import Pool
import os

os.system("taskset -p 0xff %d" % os.getpid())


def frame_avg(img):
    scaled = img.astype('uint32')
    squared = scaled**2
    avgsq = np.average(squared, axis=1)
    return np.sqrt(avgsq).astype('uint8')


def movie_iter(movie_name, frames_to_skip):
    movie = cv2.VideoCapture(movie_name)
    s, f = movie.read()
    while s:
        yield f
        for i in range(frames_to_skip):
            movie.read()
        s, f = movie.read()


def elab(movie_it):
    with ProcessPoolExecutor() as p:
        out = p.map(frame_avg, movie_it, chunksize=100)

    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("No arg!")
        exit(-1)
    start = datetime.now()
    it = movie_iter(sys.argv[1], 4)

    res = elab(it)
    print(f"Processing end: {(datetime.now() - start).total_seconds()}")
    c = np.array(res)
    cc = c.swapaxes(0, 1)
    i = Image.fromarray(cc, mode='RGB')
    i.save("prova_parall.jpg")
