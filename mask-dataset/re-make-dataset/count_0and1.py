fp = open('result-nospace.txt','r')
num_0 = 0
num_1 = 0
for eachline in fp:
    print(eachline)
    if eachline=='0\n':
        num_0=num_0+1
    if eachline=='1\n':
        num_1=num_1+1
print('num0：',num_0)
print('num1：',num_1)
