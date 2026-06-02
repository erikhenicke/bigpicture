"""
Generate presentation diagrams showing FMoW distribution shift:
  1. TVD and average per-class change per region (bar chart)
  2. Top-5 Africa classes: train vs test distribution (grouped bar chart)
  3. Sample proportion per region: in-domain vs out-of-domain (grouped bar chart)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from wilds import get_dataset

REGIONS = ["Asia", "Europe", "Africa", "Americas", "Oceania"]
CATEGORIES = [
    "airport", "airport_hangar", "airport_terminal", "amusement_park",
    "aquaculture", "archaeological_site", "barn", "border_checkpoint",
    "burial_site", "car_dealership", "construction_site", "crop_field",
    "dam", "debris_or_rubble", "educational_institution", "electric_substation",
    "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
    "gas_station", "golf_course", "ground_transportation_station", "helipad",
    "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
    "lighthouse", "military_facility", "multi-unit_residential",
    "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
    "parking_lot_or_garage", "place_of_worship", "police_station", "port",
    "prison", "race_track", "railway_bridge", "recreational_facility",
    "road_bridge", "runway", "shipyard", "shopping_mall",
    "single-unit_residential", "smokestack", "solar_farm", "space_facility",
    "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
    "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
    "wind_farm", "zoo",
]
REGION_MAP = ["Asia", "Europe", "Africa", "Americas", "Oceania", "Other"]


def load_data():
    ds = get_dataset(dataset="fmow", root_dir="data", download=False)
    labels = np.asarray(ds._y_array)
    splits = np.asarray(ds._split_array)
    metadata = np.asarray(ds._metadata_array)
    region_ids = metadata[:, 0].astype(int)
    n_classes = ds._n_classes
    return labels, splits, region_ids, n_classes


def compute_distributions(labels, splits, region_ids, n_classes):
    """Compute per-region class distributions for train (split=0) and OOD test (split=4)."""
    train_mask = splits == 0
    test_mask = splits == 4
    results = {}
    for region_name in REGIONS:
        rid = REGION_MAP.index(region_name)
        region_mask = region_ids == rid

        train_labels = labels[train_mask & region_mask]
        test_labels = labels[test_mask & region_mask]

        train_dist = np.bincount(train_labels, minlength=n_classes).astype(float)
        test_dist = np.bincount(test_labels, minlength=n_classes).astype(float)

        train_dist = train_dist / train_dist.sum() * 100
        test_dist = test_dist / test_dist.sum() * 100

        diff = np.abs(train_dist - test_dist)
        tvd = diff.sum() / 2
        avg_change = diff.mean()

        results[region_name] = {
            "train": train_dist,
            "test": test_dist,
            "tvd": tvd,
            "avg_change": avg_change,
        }
    return results


def plot_tvd_summary(results):
    """Diagram 1: TVD per region."""
    regions = list(results.keys())
    tvds = [results[r]["tvd"] for r in regions]

    order = np.argsort(tvds)[::-1]
    regions = [regions[i] for i in order]
    tvds = [tvds[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    x = np.arange(len(regions))
    w = 0.5

    bars = ax.bar(x, tvds, w, color="#4c97b6", edgecolor="white", zorder=3)
    ax.set_ylabel("Total Variation Distance (%)")
    ax.set_ylim(0, max(tvds) * 1.25)

    for bar, val in zip(bars, tvds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle("Label Shift per Region\n(Train vs OOD Test)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("fmow_tvd_per_region.pdf", bbox_inches="tight")
    fig.savefig("fmow_tvd_per_region.png", bbox_inches="tight", dpi=200)
    print("Saved fmow_tvd_per_region.pdf / .png")
    plt.close(fig)


def plot_africa_top5(results):
    """Diagram 2: Top-5 most common Africa classes, train vs test."""
    train_dist = results["Africa"]["train"]
    test_dist = results["Africa"]["test"]

    top5_idx = np.argsort(train_dist)[::-1][:5]

    class_names = [CATEGORIES[i].replace("_", " ").title() for i in top5_idx]
    train_vals = train_dist[top5_idx]
    test_vals = test_dist[top5_idx]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(class_names))
    w = 0.32

    bars_train = ax.bar(x - w / 2, train_vals, w, label="Train", color="#4c97b6", edgecolor="white", zorder=3)
    bars_test = ax.bar(x + w / 2, test_vals, w, label="OOD Test", color="#e85661", edgecolor="white", zorder=3)

    for bar, val in zip(bars_train, train_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#4c97b6", fontweight="bold")
    for bar, val in zip(bars_test, test_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#e85661", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_ylabel("Share of Samples (%)")
    ax.set_ylim(0, max(max(train_vals), max(test_vals)) * 1.3)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(framealpha=0.9, fontsize=10)

    fig.suptitle("Africa: Top 5 Classes\n(Train vs OOD Test)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95)
    fig.savefig("fmow_africa_top5.pdf", bbox_inches="tight")
    fig.savefig("fmow_africa_top5.png", bbox_inches="tight", dpi=200)
    print("Saved fmow_africa_top5.pdf / .png")
    plt.close(fig)


def plot_region_proportions():
    """Diagram 3: Sample proportion per region, in-domain vs out-of-domain."""
    id_counts = {"Asia": 23117, "Europe": 45234, "Africa": 1981, "Americas": 27179, "Oceania": 2110}
    ood_counts = {"Asia": 9084, "Europe": 13590, "Africa": 3396, "Americas": 14586, "Oceania": 1359}

    id_total = sum(id_counts.values())
    ood_total = sum(ood_counts.values())

    regions = REGIONS
    id_pct = [id_counts[r] / id_total * 100 for r in regions]
    ood_pct = [ood_counts[r] / ood_total * 100 for r in regions]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(regions))
    w = 0.32

    bars_id = ax.bar(x - w / 2, id_pct, w, label="In-Domain (before 2013)", color="#4c97b6", edgecolor="white", zorder=3)
    bars_ood = ax.bar(x + w / 2, ood_pct, w, label="Out-of-Domain (after 2013)", color="#e85661", edgecolor="white", zorder=3)

    for bar, val in zip(bars_id, id_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#4c97b6", fontweight="bold")
    for bar, val in zip(bars_ood, ood_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#e85661", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=11)
    ax.set_ylabel("Share of Samples (%)")
    ax.set_ylim(0, max(max(id_pct), max(ood_pct)) * 1.25)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(framealpha=0.9, fontsize=10)

    fig.suptitle("Sample Distribution per Region", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("fmow_region_proportions.pdf", bbox_inches="tight")
    fig.savefig("fmow_region_proportions.png", bbox_inches="tight", dpi=200)
    print("Saved fmow_region_proportions.pdf / .png")
    plt.close(fig)


if __name__ == "__main__":
    labels, splits, region_ids, n_classes = load_data()
    print(f"Loaded data: {len(labels)} samples, {n_classes} classes")
    results = compute_distributions(labels, splits, region_ids, n_classes)

    for r in REGIONS:
        print(f"{r:10s}  TVD: {results[r]['tvd']:5.1f}%  Avg Change: {results[r]['avg_change']:.2f}%")

    plot_tvd_summary(results)
    plot_africa_top5(results)
    plot_region_proportions()
