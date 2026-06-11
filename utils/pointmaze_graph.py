"""Conservative dataset-position graph utilities for PointMaze diagnostics."""

from collections import deque
import json

import numpy as np


def parse_xy_dims(value):
    """Parse an xy dim flag such as '0,1'."""
    if isinstance(value, str):
        dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        dims = tuple(int(part) for part in value)
    if len(dims) != 2:
        raise ValueError(f"Expected exactly two xy dims, got {dims}.")
    return dims


def dataset_xy(dataset, xy_dims=(0, 1)):
    """Return xy positions from a Dataset-like object."""
    xy_dims = parse_xy_dims(xy_dims)
    observations = np.asarray(dataset["observations"])
    if observations.ndim != 2:
        raise ValueError(f"Expected 2D observations, got shape {observations.shape}.")
    if max(xy_dims) >= observations.shape[-1]:
        raise ValueError(
            f"xy dims {xy_dims} exceed observation dim {observations.shape[-1]}."
        )
    return observations[:, xy_dims].astype(np.float32)


def valid_transition_indices(dataset):
    """Return indices whose next observation belongs to the same trajectory."""
    size = int(dataset.size if hasattr(dataset, "size") else len(dataset["observations"]))
    if "valids" in dataset:
        valid = np.asarray(dataset["valids"]) > 0
        idxs = np.nonzero(valid)[0]
    elif "terminals" in dataset:
        terminals = np.asarray(dataset["terminals"]) > 0
        idxs = np.nonzero(~terminals)[0]
    else:
        idxs = np.arange(size - 1)
    return idxs[idxs < size - 1].astype(np.int32)


def source_indices(dataset):
    """Return valid source-state indices for diagnostic pair sampling."""
    if hasattr(dataset, "valid_idxs"):
        idxs = np.asarray(dataset.valid_idxs, dtype=np.int32)
    else:
        idxs = valid_transition_indices(dataset)
    return idxs[idxs < len(dataset["observations"])].astype(np.int32)


def median_step_xy(datasets, xy_dims=(0, 1), max_samples=200000):
    """Estimate the median nonzero one-step xy displacement."""
    deltas = []
    for dataset in datasets:
        xy = dataset_xy(dataset, xy_dims)
        idxs = valid_transition_indices(dataset)
        if len(idxs) == 0:
            continue
        if len(idxs) > max_samples:
            idxs = np.linspace(0, len(idxs) - 1, max_samples).astype(np.int64)
            idxs = valid_transition_indices(dataset)[idxs]
        step = np.linalg.norm(xy[idxs + 1] - xy[idxs], axis=-1)
        step = step[step > 1e-8]
        if len(step) > 0:
            deltas.append(step)
    if not deltas:
        raise ValueError("No nonzero valid xy transitions found.")
    return float(np.median(np.concatenate(deltas)))


def _bin_xy(xy, origin, bin_size):
    return np.floor((xy - origin[None, :]) / float(bin_size)).astype(np.int32)


def _edge_pairs_for_dataset(dataset, state_to_bin):
    idxs = valid_transition_indices(dataset)
    src = state_to_bin[idxs]
    dst = state_to_bin[idxs + 1]
    mask = src != dst
    if not mask.any():
        return np.zeros((0, 2), dtype=np.int32)
    src = src[mask]
    dst = dst[mask]
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    return np.stack([lo, hi], axis=-1).astype(np.int32)


def build_dataset_position_graph(
    train_dataset,
    val_dataset,
    xy_dims=(0, 1),
    bin_size=None,
    bin_size_factor=2.0,
):
    """Build an undirected graph from observed dataset transitions.

    Nodes are occupied xy bins from train+validation observations. Edges are
    only consecutive observed transitions, so this avoids geometric shortcuts
    through walls.
    """
    xy_dims = parse_xy_dims(xy_dims)
    train_xy = dataset_xy(train_dataset, xy_dims)
    val_xy = dataset_xy(val_dataset, xy_dims)
    median_step = median_step_xy((train_dataset, val_dataset), xy_dims)
    if bin_size is None:
        bin_size = max(1e-6, float(bin_size_factor) * median_step)
    bin_size = float(bin_size)

    all_xy = np.concatenate([train_xy, val_xy], axis=0)
    origin = np.floor(all_xy.min(axis=0) / bin_size) * bin_size
    all_coords = _bin_xy(all_xy, origin, bin_size)
    unique_coords, inverse = np.unique(all_coords, axis=0, return_inverse=True)
    train_state_to_bin = inverse[: len(train_xy)].astype(np.int32)
    val_state_to_bin = inverse[len(train_xy) :].astype(np.int32)
    bin_centers = origin[None, :] + (unique_coords.astype(np.float32) + 0.5) * bin_size

    edge_pairs = np.concatenate(
        [
            _edge_pairs_for_dataset(train_dataset, train_state_to_bin),
            _edge_pairs_for_dataset(val_dataset, val_state_to_bin),
        ],
        axis=0,
    )
    if len(edge_pairs) > 0:
        edge_pairs = np.unique(edge_pairs, axis=0)

    metadata = dict(
        xy_dims=list(xy_dims),
        bin_size=bin_size,
        median_step_xy=median_step,
        bin_size_factor=float(bin_size_factor),
        env_steps_per_graph_edge=max(1.0, bin_size / max(median_step, 1e-6)),
        graph_kind="dataset_position_observed_transition",
    )
    return dict(
        bin_centers=bin_centers.astype(np.float32),
        bin_coords=unique_coords.astype(np.int32),
        edge_src=edge_pairs[:, 0].astype(np.int32),
        edge_dst=edge_pairs[:, 1].astype(np.int32),
        train_state_to_bin=train_state_to_bin,
        val_state_to_bin=val_state_to_bin,
        metadata=metadata,
    )


def save_graph_npz(path, graph):
    """Save a graph dictionary to an npz file."""
    np.savez_compressed(
        path,
        bin_centers=graph["bin_centers"],
        bin_coords=graph["bin_coords"],
        edge_src=graph["edge_src"],
        edge_dst=graph["edge_dst"],
        train_state_to_bin=graph["train_state_to_bin"],
        val_state_to_bin=graph["val_state_to_bin"],
        metadata_json=np.asarray(json.dumps(graph["metadata"])),
    )


def load_graph_npz(path):
    """Load a graph dictionary saved by save_graph_npz."""
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    return dict(
        bin_centers=data["bin_centers"].astype(np.float32),
        bin_coords=data["bin_coords"].astype(np.int32),
        edge_src=data["edge_src"].astype(np.int32),
        edge_dst=data["edge_dst"].astype(np.int32),
        train_state_to_bin=data["train_state_to_bin"].astype(np.int32),
        val_state_to_bin=data["val_state_to_bin"].astype(np.int32),
        metadata=metadata,
    )


def adjacency_lists(num_nodes, edge_src, edge_dst):
    """Create undirected adjacency lists from edge arrays."""
    neighbors = [[] for _ in range(int(num_nodes))]
    for src, dst in zip(np.asarray(edge_src), np.asarray(edge_dst)):
        src = int(src)
        dst = int(dst)
        neighbors[src].append(dst)
        neighbors[dst].append(src)
    return [np.asarray(sorted(set(items)), dtype=np.int32) for items in neighbors]


def shortest_hop_distances(adjacency, source):
    """Return BFS hop distances from source; unreachable nodes are -1."""
    distances = np.full(len(adjacency), -1, dtype=np.int32)
    source = int(source)
    distances[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        next_distance = distances[node] + 1
        for neighbor in adjacency[node]:
            neighbor = int(neighbor)
            if distances[neighbor] >= 0:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
    return distances


def connected_component_sizes(adjacency):
    """Return connected component sizes for a graph."""
    seen = np.zeros(len(adjacency), dtype=bool)
    sizes = []
    for start in range(len(adjacency)):
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in adjacency[node]:
                neighbor = int(neighbor)
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        sizes.append(size)
    return np.asarray(sorted(sizes, reverse=True), dtype=np.int32)


def graph_distance_statistics(adjacency, graph=None):
    """Return all-source BFS distance statistics for a small/medium graph."""
    max_hops = 0
    finite_pair_count = 0
    hop_sum = 0.0
    for source in range(len(adjacency)):
        distances = shortest_hop_distances(adjacency, source)
        finite = distances >= 0
        if finite.any():
            max_hops = max(max_hops, int(distances[finite].max()))
            finite_pair_count += int(finite.sum())
            hop_sum += float(distances[finite].sum())
    mean_hops = hop_sum / finite_pair_count if finite_pair_count else np.nan
    scale = 1.0
    if graph is not None:
        scale = float(graph["metadata"].get("env_steps_per_graph_edge", 1.0))
    return dict(
        max_hops=int(max_hops),
        max_steps=float(max_hops * scale),
        mean_hops=float(mean_hops),
        mean_steps=float(mean_hops * scale) if np.isfinite(mean_hops) else np.nan,
        finite_pair_count=int(finite_pair_count),
    )


def bin_to_state_indices(state_to_bin, num_bins, valid_idxs=None):
    """Return state-index arrays for every bin."""
    state_to_bin = np.asarray(state_to_bin, dtype=np.int32)
    if valid_idxs is None:
        idxs = np.arange(len(state_to_bin), dtype=np.int32)
    else:
        idxs = np.asarray(valid_idxs, dtype=np.int32)
    bins = state_to_bin[idxs]
    result = [[] for _ in range(int(num_bins))]
    for idx, bin_idx in zip(idxs, bins):
        result[int(bin_idx)].append(int(idx))
    return [np.asarray(items, dtype=np.int32) for items in result]


def graph_step_distances(hop_distances, graph):
    """Convert graph hop distances to calibrated environment-step distances."""
    scale = float(graph["metadata"].get("env_steps_per_graph_edge", 1.0))
    hop_distances = np.asarray(hop_distances)
    distances = hop_distances.astype(np.float32) * scale
    distances[hop_distances < 0] = np.inf
    return distances


def sample_graph_budget_pairs(
    dataset,
    state_to_bin,
    graph,
    budget,
    num_pairs,
    rng,
    pos_boundary_frac=0.5,
    neg_max_factor=2.0,
    adjacency=None,
):
    """Sample balanced graph-distance positive/negative pairs for one budget."""
    budget = int(budget)
    rng = np.random.default_rng() if rng is None else rng
    adjacency = (
        adjacency
        if adjacency is not None
        else adjacency_lists(len(graph["bin_centers"]), graph["edge_src"], graph["edge_dst"])
    )
    state_to_bin = np.asarray(state_to_bin, dtype=np.int32)
    src_idxs = source_indices(dataset)
    src_idxs = src_idxs[state_to_bin[src_idxs] >= 0]
    goal_by_bin = bin_to_state_indices(state_to_bin, len(graph["bin_centers"]))
    has_goal = np.asarray([len(items) > 0 for items in goal_by_bin])
    distance_cache = {}

    observations = []
    actions = []
    goals = []
    budgets = []
    labels = []
    graph_distances = []
    source_bins = []
    goal_bins = []

    def distances_for_bin(bin_idx):
        bin_idx = int(bin_idx)
        if bin_idx not in distance_cache:
            hops = shortest_hop_distances(adjacency, bin_idx)
            distance_cache[bin_idx] = graph_step_distances(hops, graph)
        return distance_cache[bin_idx]

    def add_pairs(target_label, target_count):
        attempts = 0
        max_attempts = max(1000, target_count * 100)
        while target_count > 0 and attempts < max_attempts:
            attempts += 1
            src_idx = int(rng.choice(src_idxs))
            src_bin = int(state_to_bin[src_idx])
            distances = distances_for_bin(src_bin)
            finite = np.isfinite(distances) & has_goal
            if target_label == 1.0:
                lo = max(0.0, float(pos_boundary_frac) * budget)
                hi = float(budget)
                candidate_mask = finite & (distances >= lo) & (distances <= hi)
                if not candidate_mask.any() and lo > 0.0:
                    candidate_mask = finite & (distances <= hi)
            else:
                lo = np.nextafter(float(budget), np.inf)
                hi = float(neg_max_factor) * budget
                candidate_mask = finite & (distances >= lo) & (distances <= hi)
                if not candidate_mask.any():
                    candidate_mask = finite & (distances > float(budget))

            candidate_bins = np.nonzero(candidate_mask)[0]
            if len(candidate_bins) == 0:
                continue
            goal_bin = int(rng.choice(candidate_bins))
            goal_idx = int(rng.choice(goal_by_bin[goal_bin]))
            observations.append(np.asarray(dataset["observations"])[src_idx])
            actions.append(np.asarray(dataset["actions"])[src_idx])
            goals.append(np.asarray(dataset["observations"])[goal_idx])
            budgets.append(budget)
            labels.append(target_label)
            graph_distances.append(float(distances[goal_bin]))
            source_bins.append(src_bin)
            goal_bins.append(goal_bin)
            target_count -= 1

    num_pos = int(num_pairs) // 2
    num_neg = int(num_pairs) - num_pos
    add_pairs(1.0, num_pos)
    add_pairs(0.0, num_neg)

    if len(labels) == 0:
        return None
    return dict(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        goals=np.asarray(goals, dtype=np.float32),
        budgets=np.asarray(budgets, dtype=np.int32),
        labels=np.asarray(labels, dtype=np.float32),
        graph_distances=np.asarray(graph_distances, dtype=np.float32),
        source_bins=np.asarray(source_bins, dtype=np.int32),
        goal_bins=np.asarray(goal_bins, dtype=np.int32),
    )
