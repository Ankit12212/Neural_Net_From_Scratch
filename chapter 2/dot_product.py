a = [1,2,3] 
b = [2,3,4] 

dot = 0

for i, j in zip(a,b):
    dot += i*j

print(dot)