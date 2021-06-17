Parallel generator using multiprocessing.

# Usage

Please see `test.py` for examples. `test.py` requires Tensorflow, but the helper itself does not.

Notes:

- Results are returned in non-determistic order.

- Function and results must be picklable.

- The `iterable` does not have to be indexable, and it can be stateful, e.g. a generator parsing files, but it must be re-iterable if `n_epochs != 1`.

# Version `1` vs version `2`

In short, version `1` is a simplified version to minimize changes in existing code, e.g. no need to use a queue explicitly.

- Version `1` uses a global process pool with its number of workers initialized by the first call (later calls have no effect), while version `2` uses a per-generator pool to avoid resource competition between different functions. Also avoid deadlock when `max_queue` is reached.
- Version `1` returns results as function return values, while version `2` pushes results into the `parallel_map_queue`. The `parallel_map_queue` avoids large non-picklable objects.
- Version `1` unpacks multiple outputs per input (returned as a list) with `yield_flat=True`, while version `2` pushes results into a `parallel_map_queue`.
- Version `2` has `run_parallel` helper function for parallel execution without return values.

# spawn vs fork

Use `spawn` whenever possible. Need to add `multiprocessing.freeze_support()` if using `fork`. Tensorflow may crash in eager mode with `fork`.

# Implementation notes

`end_of_seq` must be `None` or a magic value if using `multiprocessing.Queue`. Other non-singleton objects, e.g. `object()`, are copied and thus cannot be compared with `is` operator. `queue.Queue` stores the reference instead.

Must check worker completion with `semaphore.get_value()` before sending the `end_of_seq` signal.

`semaphore` must be used in the main process.

Callbacks, e.g. `on_error` and `on_finish`, are executed on the main process, and thus they share address space.

Nest the `producer` function and its thread into the generator to guarantee proper lifetime management.

# License

MIT

# Credit

Inspired by and derived from

<https://gist.github.com/isarandi/fb65138c66fa61218e0bce827cb30127>

<https://gist.github.com/isarandi/a72b3e5c1b1d3e40eb857a01d91926f9>

Special thanks to István Sárándi ([isarandi](https://github.com/isarandi)).
