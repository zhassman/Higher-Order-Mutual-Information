import json
import matplotlib.pyplot as plt

colors = {
    "10":  "#1f77b4",
    "50":  "#ff7f0e",
    "100": "#2ca02c",
}

with open("samples_errors.json", "r") as f:
    data = json.load(f)

sizes = data["sizes"]
errors = data["errors"]

plt.figure()

for k, curve in errors.items():
    k_str = str(k)
    color = colors.get(k_str, "#000000") 
    plt.plot(sizes, curve, color=color, label=f"k={k}")
    plt.scatter(sizes, curve, color=color, s=7)

plt.xlabel("sample size")
plt.ylabel("relative error")
plt.title("Relative Error vs Sample Size for Different k")
plt.legend()
plt.tight_layout()
plt.savefig("relative_error_samples.png", dpi=300)
plt.close()
