import numpy as np

target = np.random.randint(10, size=10)
print(target)
predictions = np.random.randint(10, size=10)
print(predictions)

mse = np.mean((target - predictions)**2)

print(mse)

for letter in 'Python':
   if letter == 'h':
      pass
      print('This is pass block')
   print('Current Letter :', letter)