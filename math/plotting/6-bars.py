#!/usr/bin/env python3
"""Stacked bar chart of fruit per person"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ["Farrah", "Fred", "Felicia"]
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
labels = ['apples', 'bananas', 'oranges', 'peaches']

bottom = np.zeros(3)

plt.bar(people, fruit[0], 0.5, color=colors[0], label=labels[0], bottom=bottom)
bottom += fruit[0]

plt.bar(people, fruit[1], 0.5, color=colors[1], label=labels[1], bottom=bottom)
bottom += fruit[1]

plt.bar(people, fruit[2], 0.5, color=colors[2], label=labels[2], bottom=bottom)
bottom += fruit[2]

plt.bar(people, fruit[3], 0.5, color=colors[3], label=labels[3], bottom=bottom)

plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)
plt.yticks(np.arange(0, 81, 10))
plt.title('Number of Fruit per Person')
plt.legend()
plt.show()
