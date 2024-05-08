import time
from methods.training import *

# This script tests how fast API requests can be made without throwing an exception

# The optimal wait time will depend on the complexity of the prompt (time to get response),
# so these results should only be taken as a frame of refference.

count = "Give the number that comes after the input number and nothing else."
model = build_llm(count)

prompt = "0"
i = 0
x = 0

t0 = time.time()
while(i<50):
    time.sleep(3)
    try:
        prompt = model.generate_content([prompt]).text
        x += 1
    except:
        pass
    i += 1
delta = time.time() - t0

print("Number of responses: " + prompt)
print("Time elapsed: " + str(round(delta, 2)) + "s")
print("Requests per minute: " + str(round(x*60/delta, 2)))

'''
10s -> 50, 5.7/min
9s -> 47, 5.94/min *
8s -> 44, 6.25/min
7s -> 40, 6.45/min
6s -> 36, 6.93/min
5s -> 27, 6.08/min
4s -> 24, 6.71/min
3s -> 19, 6.94/min
'''