import multiprocessing as mp

output = mp.Queue()
import numpy as np


def f(a, b, pos, c=0):
    s = a+b+c
    output.put((pos, s, b, np.array([5])))

# Setup a list of processes that we want to run
print('1')
processes = [mp.Process(target=f, kwargs={'a':2*x, 'b':1, 'pos':x}) for x in range(4)]
print('2')
# Run processes
for p in processes:
    p.start()
print('3')
# Exit the completed processes
for p in processes:
    p.join()
print('4')
print(output)
# Get process results from the output queue
results = [output.get() for p in processes]
print('5')
print(results)