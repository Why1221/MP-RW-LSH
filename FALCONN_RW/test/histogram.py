import sys
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle("Distribution Histogram of Range-Summable")

with open("feigen.txt", "r") as f:
    ipt = f.readlines()
    data1 = [float(x) for x in ipt]

with open("dyasim.txt", "r") as f:
    ipt = f.readlines()
    data2 = [float(x) for x in ipt]

with open("gauss.txt", "r") as f:
    ipt = f.readlines()
    data3 = [float(x) for x in ipt]

ax1.hist(data1, bins=200)
ax1.set_title("Feigenbaum")
ax1.set_xlim([-3000,3000])
ax2.hist(data2, bins=200)
ax2.set_title("Dyadic Simulation")
ax2.set_xlim([-3000,3000])
ax3.hist(data3, bins=200)
ax3.set_title("Reference Gaussian")
ax3.set_xlim([-3000,3000])
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("histogram.png", dpi = 600)
