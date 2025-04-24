

print("hi")

x = 5

def add(a, b):
    return a + b

print(add(x, 4))

#Print Numbers 1 to 100 Write a program that prints numbers from 1 to 100.
for i in range(5):
    print(i)


#Even or Odd Ask the user for a number and print whether it’s even or odd.
n = int(input("Enter a number: "))
m = int(input("Enter a number: "))

if n % 2 == 0:
    print("even")
else:
    print("odd")

#Maximum of Three Numbers Take three integers as input and print the largest one.
a = int(input("Enter a number: "))
b = int(input("Enter a number: "))
c = int(input("Enter a number: "))

if a < c and b < c:
    print(c)
elif a < b and c < b:
    print(b)
else:
    print(a)

#Multiplication Table Ask the user for a number and print its multiplication table up to 10.
d = int(input("Enter a number: "))

for i in range(2, 9, 2): 
    print(i * d)

#Sum of Digits Take an integer input and calculate the sum of its digits.

e = int(input("Enter a number: "))

total = 0

while e // 1 != 0:
    total += e % 10
    e = e // 10

print(total)

print("hi")


