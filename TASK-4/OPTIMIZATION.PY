from pulp import LpMaximize, LpProblem, LpVariable, LpStatus, value
import matplotlib.pyplot as plt
import numpy as np


model = LpProblem("Maximize_Profit", LpMaximize)


tables = LpVariable("Tables", lowBound=0, cat='Continuous')
chairs = LpVariable("Chairs", lowBound=0, cat='Continuous')


model += 30 * tables + 20 * chairs, "Total_Profit"


model += 2 * tables + 1 * chairs <= 100, "Labor_Constraint"
model += 1 * tables + 2 * chairs <= 80, "Wood_Constraint"


model.solve()


print(f"Status: {LpStatus[model.status]}")
print(f"Optimal number of Tables: {tables.varValue}")
print(f"Optimal number of Chairs: {chairs.varValue}")
print(f"Maximum Profit: ₹{value(model.objective)}")


x = np.linspace(0, 60, 400)
y1 = 100 - 2 * x      
y2 = (80 - x) / 2     

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label=r'2x + y ≤ 100 (Labor)', color='blue')
plt.plot(x, y2, label=r'x + 2y ≤ 80 (Wood)', color='green')
plt.xlim((0, 60))
plt.ylim((0, 60))
plt.xlabel('Tables')
plt.ylabel('Chairs')
plt.title('Feasible Region and Optimal Solution')
plt.fill_between(x, 0, np.minimum(y1, y2), color='lightgray', alpha=0.5)


plt.plot(tables.varValue, chairs.varValue, 'ro', label='Optimal Solution')
plt.legend()
plt.grid(True)
plt.show()
