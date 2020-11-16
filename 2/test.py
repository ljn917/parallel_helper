import tensorflow as tf
import numpy as np

import parallel_helper


def get_dataset(n=10000):
    return tf.data.Dataset.range(n)


def get_one_batch(ds):
    return next(iter(ds))


def dataset_prepare(x, parallel_map_queue=None):
    # res = [x, x*x] # require spawn for tf.Tensor ops
    res = [x.numpy(), x.numpy()*x.numpy()]
    if parallel_map_queue is None:
        return res
    else:
        parallel_map_queue.put(res)
        return None


def dataset_prepare2(x, **kwargs):
    # res = [x, x*x] # require spawn for tf.Tensor ops
    print([x.numpy(), x.numpy()*x.numpy()])


def transform_dataset(dataset, batch_size=1, max_queue=16, n_workers=4, ctx_new_thread_method='fork'):
    _g = parallel_helper.parallel_map(
        dataset_prepare,
        dataset,
        extra_kwargs={},
        n_workers=n_workers,
        n_epochs=1,
        max_queue=max_queue,
        ctx_new_thread_method=ctx_new_thread_method,
    )
    output_types  = (tf.int64, tf.int64)
    output_shapes = (tf.TensorShape([]), tf.TensorShape([]))
    ds = tf.data.Dataset.from_generator(
        _g,
        output_types=output_types,
        output_shapes=output_shapes,
    )
    ds = ds.batch(batch_size=batch_size, drop_remainder=False)
    ds = ds.prefetch(8)
    return ds


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    print('tf version: ', tf.__version__)
    ds = get_dataset()
    
    for i in ds.take(5):
        print(i)
    
    ds1 = transform_dataset(ds)
    print(get_one_batch(ds1))
    
    n = 5
    cnt = 0
    ds5 = transform_dataset(ds.take(n))
    print('parallel transform_dataset count 5:')
    for i in ds5:
        print(i)
        cnt += 1
    assert cnt == n
    print('end parallel transform_dataset count 5')
    
    def _iterate_test(n=1000, ctx_new_thread_method='fork'):
        n = 1000
        ds = get_dataset(n)
        ds = transform_dataset(ds, ctx_new_thread_method=ctx_new_thread_method)
        cnt = 0
        for i in ds:
            print(i)
            cnt += 1
        print('cnt: ', cnt)
        assert cnt == n
    
    _iterate_test(n=1000, ctx_new_thread_method='fork')
    _iterate_test(n=1000, ctx_new_thread_method='spawn')
    
    n = 1000
    ds = get_dataset(n)
    parallel_helper.run_parallel(dataset_prepare2, ds, 4, extra_args=None, wait_time=120, ctx_new_thread_method='fork')
    parallel_helper.run_parallel(dataset_prepare2, ds, 4, extra_args=None, wait_time=120, ctx_new_thread_method='spawn')
