# Inline Question 4 — Detailed Solution

## Setup

We are given f(x, y) = xy and asked two things:

1. Evaluate min_x max_y f(x, y)
2. Perform 6 steps of alternating gradient descent/ascent starting from (x₀, y₀) = (1, 1) with learning rate η = 1

---

## Part 1: Evaluating min_x max_y xy

We need to find: min_x [ max_y xy ]

Consider three cases for the inner maximization over y:

**Case 1: x > 0**
- xy grows without bound as y → +∞
- max_y xy = +∞

**Case 2: x = 0**
- xy = 0 for all y
- max_y xy = 0

**Case 3: x < 0**
- xy grows without bound as y → -∞
- max_y xy = +∞

So the inner max is:

```
max_y xy = { +∞   if x ≠ 0
           {  0   if x = 0
```

Taking the outer min over x, the only finite value is at x = 0:

**min_x max_y xy = 0**

---

## Part 2: Alternating Gradient Updates

### Gradients

```
f(x, y) = xy

∂f/∂y = x
∂f/∂x = y
```

### Update Rules (η = 1)

y is the maximizer → gradient **ascent**:

```
y_i = y_{i-1} + η · (∂f/∂y) evaluated at (x_{i-1}, y_{i-1})
y_i = y_{i-1} + x_{i-1}
```

x is the minimizer → gradient **descent** (using the **updated** y_i):

```
x_i = x_{i-1} - η · (∂f/∂x) evaluated at (x_{i-1}, y_i)
x_i = x_{i-1} - y_i
```

### Step-by-Step Computation

**Step 0 (initial):**
```
x₀ = 1,  y₀ = 1
f(x₀, y₀) = 1 · 1 = 1
```

**Step 1:**
```
y₁ = y₀ + x₀ = 1 + 1 = 2
x₁ = x₀ - y₁ = 1 - 2 = -1
f(x₁, y₁) = (-1)(2) = -2
```

**Step 2:**
```
y₂ = y₁ + x₁ = 2 + (-1) = 1
x₂ = x₁ - y₂ = -1 - 1 = -2
f(x₂, y₂) = (-2)(1) = -2
```

**Step 3:**
```
y₃ = y₂ + x₂ = 1 + (-2) = -1
x₃ = x₂ - y₃ = -2 - (-1) = -2 + 1 = -1
f(x₃, y₃) = (-1)(-1) = 1
```

**Step 4:**
```
y₄ = y₃ + x₃ = -1 + (-1) = -2
x₄ = x₃ - y₄ = -1 - (-2) = -1 + 2 = 1
f(x₄, y₄) = (1)(-2) = -2
```

**Step 5:**
```
y₅ = y₄ + x₄ = -2 + 1 = -1
x₅ = x₄ - y₅ = 1 - (-1) = 1 + 1 = 2
f(x₅, y₅) = (2)(-1) = -2
```

**Step 6:**
```
y₆ = y₅ + x₅ = -1 + 2 = 1
x₆ = x₅ - y₆ = 2 - 1 = 1
f(x₆, y₆) = (1)(1) = 1
```

### Summary Table

| Step |  y  |  x  | f(x,y) |
|:----:|:---:|:---:|:------:|
|  0   |  1  |  1  |    1   |
|  1   |  2  | -1  |   -2   |
|  2   |  1  | -2  |   -2   |
|  3   | -1  | -1  |    1   |
|  4   | -2  |  1  |   -2   |
|  5   | -1  |  2  |   -2   |
|  6   |  1  |  1  |    1   |

### Conclusion

The iterates return to (x₆, y₆) = (1, 1) = (x₀, y₀), forming a cycle of period 6. The function f(x, y) oscillates between 1 and -2, never reaching the optimal value of 0. This demonstrates that alternating gradient descent/ascent on a minimax objective can cycle indefinitely without converging, which is a fundamental challenge in GAN training.
