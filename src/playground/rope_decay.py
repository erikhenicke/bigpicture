from matplotlib import pyplot as plt

thetas = [10, 100, 1000]
colors = ['blue', 'orange', 'green', 'red']
dimensions = [5, 10, 20, 50]

def decay(theta, d):
    return [theta ** -(i / d) for i in list(range(d+1))]

for d in dimensions:
    for theta, color in zip(thetas, colors):
        plt.plot(decay(theta, d), label=f'theta={theta}, d={d}', color=color)

plt.legend()
plt.show()