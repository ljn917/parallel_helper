# Copyright (c) 2020  István Sárándi <sarandi@vision.rwth-aachen.de>
# Copyright (c) 2020-2021  @ljn917
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modified from https://gist.github.com/isarandi/a72b3e5c1b1d3e40eb857a01d91926f9
# and https://gist.github.com/isarandi/fb65138c66fa61218e0bce827cb30127

import os
import time
import threading
import multiprocessing
import queue
import itertools
import signal
import ctypes


def parallel_map(
        fun,
        iterable,
        extra_args=(),
        extra_kwargs=None,
        n_workers=4,
        max_queue=256,
        n_epochs=None,
        yield_flat=False,
        ctx_new_thread_method='spawn',
        ):
    """Maps `fun` to each element of `iterable` with multiprocessing.

    Args:
        fun: A function that takes an element from `iterable` plus `extra_args` and `extra_kwargs`.
        iterable: An iterable holding the input objects. It must be re-iterable.
        extra_args: extra positional arguments for `fun`.
        extra_kwargs: extra keyword arguments for `fun`.
        n_workers: Number of worker processes. Workers are shared if multiple generators exist.
            It would be ignored if worker pool exists.
        n_epochs: Number of times to iterate over the `iterable`.
        yield_flat: While True, unpack the return value as a list of results.

    Returns:
        A generator function
    """
    # tf.dataset.from_generator requires a generator factory
    if extra_kwargs is None:
        extra_kwargs = {}
    def _g():
        pool = get_pool(n_workers, ctx_new_thread_method=ctx_new_thread_method)
        semaphore = multiprocessing.Semaphore(n_workers)
        q = queue.Queue(max_queue)
        
        end_of_seq = object()

        if n_epochs is None:
            epoch_counter = itertools.count()
        else:
            epoch_counter = range(n_epochs)

        def producer():
            def on_error(e):
                print(e)
                semaphore.release()

            for _ in epoch_counter:
                for item in iterable:
                    semaphore.acquire()
                    pool.apply_async(fun, args=(item, *extra_args), kwds=extra_kwargs, callback=q.put, error_callback=on_error)
            # make sure all workers finished before sending None
            while(semaphore.get_value() != n_workers):
                time.sleep(1)
            q.put(end_of_seq)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        while True:
            result = q.get()
            if result is end_of_seq:
                return
            else:
                if yield_flat:
                    for i in result:
                        yield tuple(i)
                else:
                    yield tuple(result)
            semaphore.release()
    return _g

_pool = None

def get_pool(n_workers_if_uninitialized, ctx_new_thread_method='spawn'):
    global _pool
    if _pool is None:
        ctx = multiprocessing.get_context(ctx_new_thread_method)
        _pool = ctx.Pool(n_workers_if_uninitialized, initializer=init_worker_process)
    return _pool

def init_worker_process():
    terminate_on_parent_death()
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def terminate_on_parent_death():
    prctl = ctypes.CDLL("libc.so.6").prctl
    PR_SET_PDEATHSIG = 1
    prctl(PR_SET_PDEATHSIG, signal.SIGTERM) 

if __name__ == '__main__':
    multiprocessing.freeze_support()
