"""Graph analysis utilities."""
from typing import NamedTuple, Optional, Generator, Union

import numpy as np
import pandas as pd
import graph_tool.all as gt


def pair_term_graph(pair_term_df: pd.DataFrame):
    """Represents the graph in the pair term dataframe using graph-tool.

    Args:
        pair_term_df: The dataframe describing each pair term (from pairengine.py).
    Returns:
        A graph-tool Graph with vertex and edge properties describing pair term
            attributes.
    """
    vertices, edges = extract_edges(pair_term_df)

    g = gt.Graph()
    gvs = g.add_vertex(len(vertices))

    gvlookup = dict()
    types = g.new_vertex_property("string")
    layers = g.new_vertex_property("int")
    indices = g.new_vertex_property("int")
    for gv, v in zip(gvs, vertices):
        gvlookup[v] = gv

        types[gv] = v.type
        layers[gv] = v.layer
        indices[gv] = v.index

    g.vertex_properties["type"] = types
    g.vertex_properties["layer"] = layers
    g.vertex_properties["index"] = indices

    # graph metadata
    src_types = g.new_edge_property("string")
    src_layers = g.new_edge_property("int")
    src_indices = g.new_edge_property("int")
    dst_types = g.new_edge_property("string")
    dst_layers = g.new_edge_property("int")
    dst_indices = g.new_edge_property("int")
    term_types = g.new_edge_property("string")
    term_values = g.new_edge_property("float")

    for edge in edges:
        graphedge = g.add_edge(gvlookup[edge.src], gvlookup[edge.dst])

        src_types[graphedge] = edge.src.type
        src_layers[graphedge] = edge.src.layer
        src_indices[graphedge] = edge.src.index
        dst_types[graphedge] = edge.dst.type
        dst_layers[graphedge] = edge.dst.layer
        dst_indices[graphedge] = edge.dst.index
        term_types[graphedge] = edge.term_type
        term_values[graphedge] = edge.term_value

    g.edge_properties["src_type"] = src_types
    g.edge_properties["src_layer"] = src_layers
    g.edge_properties["src_indices"] = src_indices
    g.edge_properties["dst_type"] = dst_types
    g.edge_properties["dst_layer"] = dst_layers
    g.edge_properties["dst_indices"] = dst_indices
    g.edge_properties["term_type"] = term_types
    g.edge_properties["term_value"] = term_values

    return g


# Small dataclass-esque objects
Vertex = NamedTuple("Vertex", [("type", str), ("layer", int), ("index", int)])


Edge = NamedTuple(
    "Edge",
    [("src", Vertex), ("dst", Vertex), ("term_type", str), ("term_value", float)],
)


def extract_edges(pair_term_df: pd.DataFrame):
    """Extracts the edges of the pair term dataframe.

    Args:
        pair_term_df: The dataframe describing each pair term (from pairengine.py).
    Returns:
        A list of vertices and edges that describe the graph.
    """

    def vertex_to_name(vertex_type: str, layer: int, index: int) -> str:
        return f"{vertex_type}-{layer}-{index}"

    vertices = set()
    edges = list()

    for (st, sl, si, dt, dl, di, tt, tv) in zip(
        pair_term_df["src_type"],
        pair_term_df["src_layer"],
        pair_term_df["src_index"],
        pair_term_df["dst_type"],
        pair_term_df["dst_layer"],
        pair_term_df["dst_index"],
        pair_term_df["term_type"],
        pair_term_df["term_value"],
    ):
        src = Vertex(st, sl, si)
        dst = Vertex(dt, dl, di)

        vertices.add(src)
        vertices.add(dst)

        edges.append(Edge(src, dst, tt, tv))

    return vertices, edges


def filter_edges_by_arr(g: gt.Graph, edgefilter: np.ndarray) -> gt.Graph:
    """Applies a numpy arr edge filter to a graph."""
    filterprop = g.new_edge_property("bool")
    filterprop.a = edgefilter

    g.set_edge_filter(filterprop)

    return g


def filter_verts_by_arr(g: gt.Graph, vertexfilter: np.ndarray) -> gt.Graph:
    """Applies a numpy arr vertex filter to a graph."""
    filterprop = g.new_vertex_property("bool")
    filterprop.a = vertexfilter

    g.set_vertex_filter(filterprop)

    return g


def threshold_value(g: gt.Graph, threshold: float) -> gt.Graph:
    """Threshold the graph with a term_value threshold."""
    return filter_edges_by_arr(g, g.edge_properties["term_value"].a > threshold)


def single_type_graph(g: gt.Graph, term_type: str) -> gt.Graph:
    """Filter the graph to vertices of a single [src,dst]_type."""
    return filter_edges_by_arr(
        g,
        (g.edge_properties["src_type"].a == term_type)
        & (g.edge_properties["dst_type"].a == term_type),
    )


def longest_input_paths(g: gt.Graph) -> list[int]:
    """Computes the length of the longest input path to each vertex."""
    topological = gt.topological_sort(g)
    reverse = np.argsort(topological)

    longest = np.zeros(topological.shape, dtype=topological.dtype)

    # don't think I can avoid this for Python loop w/o an extension
    for (i, t) in enumerate(topological):
        prev_longest = longest[reverse[g.get_in_neighbors(t)]]
        if len(prev_longest) > 0:
            longest[i] = prev_longest.max() + 1

    return longest[reverse]


def longest_path(g: gt.Graph) -> int:
    """Computes the length of the longest input path in the graph."""
    return np.max(longest_input_paths(g))


def input_path_complexities(g: gt.Graph) -> list[float]:
    """Computes input path complexities for each vertex in the graph.

    Input path complexity is defined as the length of the longest input
    path normalized by the maximum possible length (base-0 layer index).

    Args:
        g: Graph describing pair terms.
    """
    longest_inputs = longest_input_paths(g)

    layers = g.vertex_properties["layer"].a

    layers[layers == 0] = -1

    complexities = longest_inputs / layers
    complexities[layers == -1] = -1

    return complexities


def ipc_percentiles(
    gs: Union[gt.Graph, Generator[gt.Graph, None, None]],
    percs: Optional[list[int]] = [0, 25, 50, 75, 100],
    vertex_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes IPC percentiles over all graphs in the passed generator."""

    def compute_percentiles(g):
        cs = input_path_complexities(g)

        if vertex_mask is not None:
            cs = cs[vertex_mask]
        cs = cs[cs != -1]

        return np.percentile(cs, percs)

    if isinstance(gs, gt.Graph):
        return np.array([compute_percentiles(gs)])

    ipc_percs = list()
    for g in gs:
        ipc_percs.append(compute_percentiles(g))

    return np.array(ipc_percs)


# inspired by https://stackoverflow.com/questions/20262712/enumerating-all-paths-in-a-directed-acyclic-graph  # noqa
def all_paths(g: gt.Graph, verbose: bool = True) -> list[list[int]]:
    """Finds ALL paths in a DAG.

    Often too slow to run on large models.
    """

    def dfs(
        g: gt.Graph, currpath: list[list[int]], branchpaths: list[list[int]]
    ) -> list[list[int]]:
        v = currpath[-1]

        ws = g.get_out_neighbors(v)
        if len(ws) > 0:
            for w in ws:
                branchpaths = dfs(g, currpath + [w], branchpaths)
        else:
            branchpaths += [currpath]

        return branchpaths

    all_paths = []
    num_vertices = len(g.get_vertices())
    for v in g.iter_vertices():
        if verbose:
            print(f"Starting from vertex: {v}/{num_vertices}", end="    \r")
        node_paths = dfs(g, [v], [])
        all_paths += node_paths

    return all_paths


def random_removal(g: gt.Graph) -> Generator[gt.Graph, None, None]:
    perm = np.random.permutation(np.arange(len(g.get_edges())))
    mask = np.ones((len(perm),), dtype=np.bool)

    # return the full graph first
    g.set_edge_filter(None)
    yield g

    edges_per_percentile = int(np.ceil(len(perm) / 100))
    for i in range(100):
        inds = perm[i * edges_per_percentile : (i + 1) * edges_per_percentile]
        mask[inds] = False
        filter_edges_by_arr(g, mask)

        yield g

    g.set_edge_filter(None)


def value_perc_thresholds(g: gt.Graph) -> Generator[gt.Graph, None, None]:
    """Computes IPC percentiles over all (whole) term_value percentiles."""
    value_percs = np.percentile(g.edge_properties["term_value"].a, np.arange(100) + 1)

    # return the full graph first
    g.set_edge_filter(None)
    yield g

    for v in value_percs:
        yield threshold_value(g, v)

    g.set_edge_filter(None)
