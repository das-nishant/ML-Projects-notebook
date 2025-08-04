
## 🐍 Python Beginner Tutorial

### ✅ What You'll Learn:

1. Printing to the screen
2. Variables
3. Data types
4. User input
5. If statements
6. Loops (for, while)
7. Functions
8. Lists (arrays)

---

### 1. **Hello, World!**

```python
print("Hello, world!")
```

📌 `print()` displays text or values on the screen.

---

### 2. **Variables**

```python
name = "Alice"
age = 25
height = 5.6

print(name)
print(age)
```

📌 Variables store values like text (`str`), numbers (`int`, `float`).

---

### 3. **Data Types**

```python
# String
message = "Hello"

# Integer
year = 2025

# Float
price = 19.99

# Boolean
is_raining = False

print(type(price))  # This will print: <class 'float'>
```

---

### 4. **Getting User Input**

```python
name = input("What is your name? ")
print("Hello, " + name + "!")
```

📌 `input()` takes user input as a string.

---

### 5. **If Statements**

```python
age = int(input("Enter your age: "))

if age >= 18:
    print("You're an adult!")
else:
    print("You're a minor!")
```

📌 `if`, `elif`, and `else` help your code make decisions.

---

### 6. **Loops**

#### `while` Loop

```python
count = 1
while count <= 5:
    print("Count:", count)
    count += 1
```

#### `for` Loop

```python
for i in range(5):
    print("i is", i)
```

---

### 7. **Functions**

```python
def greet(name):
    print("Hello,", name)

greet("Alice")
greet("Bob")
```

📌 Functions help you reuse blocks of code.

---

### 8. **Lists**

```python
fruits = ["apple", "banana", "cherry"]

print(fruits[0])  # apple
fruits.append("orange")
print(fruits)
```

---

## 🧠 Practice Idea: Make a Mini Quiz

```python
question = input("What is 2 + 2? ")
if question == "4":
    print("Correct!")
else:
    print("Try again.")
```
