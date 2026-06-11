#!/usr/bin/env python
"""Build a conservative dataset-position graph for PointMaze diagnostics."""

from pathlib import Path
import sys

import numpy as np
from absl import app, flags

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.env_utils import make_env_and_datasets
from utils.pointmaze_graph import (
    adjacency_lists,
    build_dataset_position_graph,
    connected_component_sizes,
    graph_distance_statistics,
    parse_xy_dims,
    save_graph_npz,
)


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "pointmaze-medium-navigate-v0", "Environment name.")
flags.DEFINE_string("dataset_dir", None, "Optional OGBench dataset directory.")
flags.DEFINE_string("output", "exp/bmm_pointmaze_graph.npz", "Output graph npz.")
flags.DEFINE_string("xy_dims", "0,1", "Comma-separated observation dims to use as xy.")
flags.DEFINE_float(
    "bin_size",
    None,
    "Optional xy bin size. If unset, use bin_size_factor * median step.",
)
flags.DEFINE_float(
    "bin_size_factor",
    2.0,
    "Bin size multiplier applied to median one-step xy displacement.",
)


def dataset_path_from_dir(dataset_dir):
    if dataset_dir is None:
        return None
    candidates = [
        str(path)
        for path in sorted(Path(dataset_dir).glob("*.npz"))
        if "-val.npz" not in path.name
    ]
    return candidates[0] if candidates else None


def main(_):
    xy_dims = parse_xy_dims(FLAGS.xy_dims)
    dataset_path = dataset_path_from_dir(FLAGS.dataset_dir)
    _, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, dataset_path=dataset_path
    )
    graph = build_dataset_position_graph(
        train_dataset,
        val_dataset,
        xy_dims=xy_dims,
        bin_size=FLAGS.bin_size,
        bin_size_factor=FLAGS.bin_size_factor,
    )
    output = Path(FLAGS.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_graph_npz(output, graph)

    adjacency = adjacency_lists(
        len(graph["bin_centers"]), graph["edge_src"], graph["edge_dst"]
    )
    component_sizes = connected_component_sizes(adjacency)
    distance_stats = graph_distance_statistics(adjacency, graph)
    metadata = graph["metadata"]
    print("Built PointMaze dataset-position graph")
    print(f"  env_name: {FLAGS.env_name}")
    print(f"  output: {output}")
    print(f"  xy_dims: {tuple(metadata['xy_dims'])}")
    print(f"  median_step_xy: {metadata['median_step_xy']:.6f}")
    print(f"  bin_size: {metadata['bin_size']:.6f}")
    print(
        "  env_steps_per_graph_edge: "
        f"{metadata['env_steps_per_graph_edge']:.6f}"
    )
    print(f"  nodes: {len(graph['bin_centers'])}")
    print(f"  undirected_edges: {len(graph['edge_src'])}")
    print(f"  components: {len(component_sizes)}")
    print(f"  largest_components: {component_sizes[:10].tolist()}")
    print(
        "  graph diameter hops/steps: "
        f"{distance_stats['max_hops']} / {distance_stats['max_steps']:.2f}"
    )
    print(
        "  mean finite distance hops/steps: "
        f"{distance_stats['mean_hops']:.2f} / {distance_stats['mean_steps']:.2f}"
    )
    degrees = np.asarray([len(items) for items in adjacency], dtype=np.int32)
    print(
        "  degree min/mean/max: "
        f"{degrees.min()} / {degrees.mean():.2f} / {degrees.max()}"
    )


if __name__ == "__main__":
    app.run(main)
