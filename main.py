import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # for drawing patches
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

# -----------------------
# PARAMETERS & SETTINGS
# -----------------------

# Map dimensions
MAP_WIDTH = 350
MAP_HEIGHT = 250

# Clusters and sites per cluster
NUM_CLUSTERS = 10
SITES_PER_CLUSTER = 20

# Weights for the objective
w_energy = 1.0
w_biodiversity = 1.0
w_distance = 1.0

# Constraints: Fixed number of selected sites OR a given total energy/biodiversity level
N_SELECTED_SITES = 40  # Fixed number of selected sites
MIN_TOTAL_ENERGY = 1500  # Minimum total energy requirement
MIN_TOTAL_BIODIVERSITY = 1500  # Minimum biodiversity requirement

# -----------------------
# HELPER FUNCTION TO GENERATE SEPARATED CLUSTER CENTERS
# -----------------------

def generate_cluster_centers(num, map_width, map_height, min_sep):
    centers = []
    attempts = 0
    max_attempts = 10000  # safeguard against infinite loop
    while len(centers) < num and attempts < max_attempts:
        # Generate candidate within 10% and 90% of the map to avoid the edges
        candidate_x = random.uniform(0.1 * map_width, 0.9 * map_width)
        candidate_y = random.uniform(0.1 * map_height, 0.9 * map_height)
        # Check if candidate is sufficiently far from all existing centers
        if all(math.hypot(candidate_x - cx, candidate_y - cy) >= min_sep for (cx, cy) in centers):
            centers.append((candidate_x, candidate_y))
        attempts += 1
    if len(centers) < num:
        raise ValueError("Could not generate enough separated centers; try reducing min_sep.")
    return centers

# -----------------------
# DATA GENERATION
# -----------------------

# For reproducibility
random.seed(42)

# Generate well-separated cluster centers
min_sep = 40  # adjust as needed
cluster_centers = generate_cluster_centers(NUM_CLUSTERS, MAP_WIDTH, MAP_HEIGHT, min_sep)

# Generate clusters and sites
clusters = {}
for c in range(NUM_CLUSTERS):
    center_x, center_y = cluster_centers[c]
    clusters[c] = {
        "center": (center_x, center_y),
        "sites": []
    }
    for i in range(SITES_PER_CLUSTER):
        # Spread sites around the cluster center
        x = center_x + random.uniform(-20, 20)
        y = center_y + random.uniform(-20, 20)
        energy = random.uniform(0, 100)
        biod = random.uniform(0, 100)
        clusters[c]["sites"].append({
            "x": x,
            "y": y,
            "energy": energy,
            "biod": biod
        })

# Compute centroids and distances for each cluster
for c in range(NUM_CLUSTERS):
    sites = clusters[c]["sites"]
    avg_x = sum(site["x"] for site in sites) / len(sites)
    avg_y = sum(site["y"] for site in sites) / len(sites)
    clusters[c]["centroid"] = (avg_x, avg_y)
    for site in sites:
        dx = site["x"] - avg_x
        dy = site["y"] - avg_y
        site["distance"] = math.sqrt(dx*dx + dy*dy)

# -----------------------
# SET UP THE LP PROBLEM
# -----------------------

# Create the optimization problem: maximize weighted benefits minus distance penalty.
prob = LpProblem("Fixed_Site_Selection", LpMaximize)

# Create decision variables: one binary variable per site.
decision_vars = {}
for c in range(NUM_CLUSTERS):
    decision_vars[c] = []
    for i in range(SITES_PER_CLUSTER):
        var = LpVariable(f"x_{c}_{i}", cat=LpBinary)
        decision_vars[c].append(var)

# Build the objective function
objective_terms = []
for c in range(NUM_CLUSTERS):
    for i, site in enumerate(clusters[c]["sites"]):
        coeff = w_energy * site["energy"] + w_biodiversity * site["biod"] - w_distance * site["distance"]
        objective_terms.append(coeff * decision_vars[c][i])
        
prob += lpSum(objective_terms), "Total_Weighted_Benefit"

# Constraint: Select exactly N_SELECTED_SITES
prob += lpSum(decision_vars[c][i] for c in range(NUM_CLUSTERS) for i in range(SITES_PER_CLUSTER)) == N_SELECTED_SITES, "Fixed_Site_Count"

# Constraints: Ensure minimum total energy and biodiversity
prob += lpSum(decision_vars[c][i] * clusters[c]["sites"][i]["energy"] for c in range(NUM_CLUSTERS) for i in range(SITES_PER_CLUSTER)) >= MIN_TOTAL_ENERGY, "Min_Energy"
prob += lpSum(decision_vars[c][i] * clusters[c]["sites"][i]["biod"] for c in range(NUM_CLUSTERS) for i in range(SITES_PER_CLUSTER)) >= MIN_TOTAL_BIODIVERSITY, "Min_Biodiversity"

# -----------------------
# SOLVE THE MODEL
# -----------------------

solver = PULP_CBC_CMD(msg=False)
prob.solve(solver)

# -----------------------
# EXTRACT THE SOLUTION
# -----------------------

selected_x, selected_y = [], []
nonselected_x, nonselected_y = [], []

for c in range(NUM_CLUSTERS):
    for i, site in enumerate(clusters[c]["sites"]):
        if decision_vars[c][i].varValue > 0.5:  # site selected
            selected_x.append(site["x"])
            selected_y.append(site["y"])
        else:
            nonselected_x.append(site["x"])
            nonselected_y.append(site["y"])

# -----------------------
# PLOT THE SOLUTION WITH OVERLAY PATCHES FOR CLUSTERS
# -----------------------

plt.figure(figsize=(8, 6))
ax = plt.gca()

# Add overlay patches (ellipses) for each cluster
for c in range(NUM_CLUSTERS):
    sites = clusters[c]["sites"]
    xs = [site["x"] for site in sites]
    ys = [site["y"] for site in sites]
    centroid = clusters[c]["centroid"]
    # Compute a width and height that cover the cluster's sites (with a scaling factor)
    width = (max(xs) - min(xs)) * 1.5
    height = (max(ys) - min(ys)) * 1.5
    ellipse = mpatches.Ellipse(centroid, width, height, color='lightgray', alpha=0.3, zorder=0)
    ax.add_patch(ellipse)

# Plot non-selected sites
plt.scatter(nonselected_x, nonselected_y, c='blue', label='Non-selected', alpha=0.6, zorder=2)
# Plot selected sites
plt.scatter(selected_x, selected_y, c='red', edgecolors='black', s=80, label='Selected', zorder=3)

plt.title("Fixed Number Site Selection with Energy & Biodiversity Constraints")
plt.xlabel("X coordinate (pixels)")
plt.ylabel("Y coordinate (pixels)")
plt.legend()
plt.grid(True)
plt.xlim(0, MAP_WIDTH)
plt.ylim(0, MAP_HEIGHT)
plt.gca().invert_yaxis()  # Mimic image coordinate systems

plt.show()

# -----------------------
# PRINT SOLUTION DETAILS
# -----------------------

selected_sites = sum(var.varValue for c in range(NUM_CLUSTERS) for var in decision_vars[c])
total_energy = sum(var.varValue * clusters[c]["sites"][i]["energy"] for c in range(NUM_CLUSTERS) for i, var in enumerate(decision_vars[c]))
total_biodiversity = sum(var.varValue * clusters[c]["sites"][i]["biod"] for c in range(NUM_CLUSTERS) for i, var in enumerate(decision_vars[c]))

print("Selected sites:", selected_sites)
print("Total energy:", total_energy)
print("Total biodiversity:", total_biodiversity)
