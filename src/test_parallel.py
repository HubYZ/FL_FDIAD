from multiprocessing import Pool
import time
import multiprocessing

def func(i):
    # time.sleep(2)
    return i * 2, i


if __name__ == "__main__":
    steps = 8
    k = 0
    # time_1 = time.time()
    # for ind in range(steps):
    #     k += func(ind)
    # print("time = {}".format(time.time() - time_1))

    pool = multiprocessing.Pool(processes=8)  # 创建4个进程
    results = []

    time_2 = time.time()
    for i in range(steps):
        results.append(pool.apply_async(func, (i,)))
    pool.close()
    pool.join()
    k = 0
    for res in results:
        k += res.get()
    print("time = {}".format(time.time() - time_2))
