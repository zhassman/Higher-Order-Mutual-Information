import json
import matplotlib.pyplot as plt


colors = {
    "100000": "#4e91ba",
    "10000":  "#a87fcf",
    "1000":   "#bd5a5c",
}

with open("k_errors.json", "r") as f:
    data = json.load(f)

sample_sizes = data["sample_sizes"]
k_values = data["k_values"]
errors = data["errors"]

plt.figure()

for N in sample_sizes:
    N_str = str(N)
    curve = errors[N_str]
    color = colors.get(N_str, "#000000")
    plt.plot(k_values, curve, color=color, label=f"N={N}")
    plt.scatter(k_values, curve, color=color, s=20)

plt.xlabel("k_neighbors")
plt.ylabel("relative error")
plt.title("Relative Error vs k_neighbors")
plt.legend()
plt.tight_layout()
plt.savefig("relative_error_vs_k.png", dpi=300)
plt.close()
