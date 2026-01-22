import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import HTML, VBox
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import Normalize
from collections import Counter


# Collection of graph plotting functions

def ensure_frame_counts(G, map_uid):
    '''  
    Helper to check our Graph's nodes have associated frame_count attributes
    which is an integer describing the associate number of frames in the trajectory
    seen with the unique contact map of the given node. Value used for node
    sizes.
    '''
    map_uid = np.asarray(map_uid, dtype=int)
    U = G.number_of_nodes()
    # If map_uid is empty, handle gracefully
    if map_uid.size == 0:
        for n in range(U): G.nodes[n]['frame_count'] = 0
        return np.zeros(U, dtype=int)
        
    counts = np.bincount(map_uid, minlength=U).astype(int)
    for n in range(U):
        G.nodes[n]['frame_count'] = int(counts[n])
    return counts

def compress_sequence(seq):
    '''  
    Given a sequence, compress so that any contiguous repeats are replaced
    with just the one value whilst preserving order.
    '''
    seq = list(seq)
    if not seq:
        return []
    out = [int(seq[0])]
    for x in seq[1:]:
        xi = int(x)
        if xi != out[-1]:
            out.append(xi)
    return out

def expand_seq_to_edges(G, node_seq, expand_jumps=True, count_multiplicity=False):
    '''
    builds the temporal edge conenctivity between nodes who's unique contact maps
    appear as frames in sequence (before or after) in the trajectory. Operates on
    the contiguity compressed sequence where contiguously repeating duplicates are 
    reduced to single occurences.

    Importantly, the expand_jumps arg specifies that, if the two nodes adjacent in
    sequence temporally are not adjacent in the contact space manifold then we still
    connect them by finding the shortest edge path along the manifold edges 
    between them.

    include_multiplicity as False specifies all edges are added only once so no 
    expression of multiplicity of connections by say visual weighting, this can be
    turned on.
    '''
    if count_multiplicity:
        counts = Counter()
    else:
        seen = set()
        out = []

    for i in range(len(node_seq)-1):
        a = int(node_seq[i])
        b = int(node_seq[i+1])
        if a == b:
            continue

        def add_edge(u, v):
            key = tuple(sorted((u, v)))
            if count_multiplicity:
                counts[key] += 1
            else:
                if key not in seen:
                    seen.add(key)
                    out.append(key)

        # direct edge
        if G.has_edge(a, b) or G.has_edge(b, a):
            add_edge(a, b)
            continue

        # expand jump
        if expand_jumps:
            try:
                path = nx.shortest_path(G, source=a, target=b)
            except nx.NetworkXNoPath:
                continue
            for j in range(len(path) - 1):
                add_edge(path[j], path[j + 1])

    return counts if count_multiplicity else out

def pairs_and_widths_from_edges(edges, pos, base_w=1.0, scale_w=1.2):
    '''
    edges may be list-of-pairs OR Counter
    returns: segs ([(p0,p1),...]), widths ([w,...])
    '''
    if isinstance(edges, Counter):
        items = list(edges.items())  # [((a,b), count), ...]
        segs = [((pos[a][0], pos[a][1]), (pos[b][0], pos[b][1])) for (a,b), _ in items]
        widths = [base_w + scale_w * np.sqrt(c) for (_, c) in items]
    else:
        # list of pairs
        segs = [((pos[a][0], pos[a][1]), (pos[b][0], pos[b][1])) for a,b in edges]
        widths = [2.0] * len(segs)  # old fixed widths
    return segs, widths

def plot_graph_static(
    G,
    map_uid,
    pos=None,
    X_emb=None,
    start_frame=0,
    expand_jumps=True,
    show_shortest=True,
    show_bg=True,
    post_fold_red=True,
    custom_paths={},
    custom_paths_colors=None,
    node_custom_color=None,
    node_custom_color_title="Frames",
    folded_node=None,
    palette='RdBu_r',
    count_multiplicity=False,
    figsize=(12, 7),
    node_size_range=(8, 48),
    title="Protein Folding Contact Topology Graph",
    unique_maps=None, # Ignored in static, kept for signature compatibility 
    unique_indices=None # Ignored in static, kept for signature compatibility
):
    '''
    Plots a static non-interactive matplotlib graph given networkx graph object G. 
    Highlights post-folded state (most occupied node) edges red and the shortest
    path edges blue. Node sizes correspond to counts in the trajectory and nodes are
    coloured by their first frame's index in the trajectory.

    pos can be provided to set node positions from say a clustering projection/embedding.
    Or X_emb can be provided instead of pos (usually when generated by clustering).
    '''
    if map_uid is None:
        raise RuntimeError("map_uid required for static plot")
    
    map_uid = np.asarray(map_uid, dtype=int)
    U = G.number_of_nodes()
    F = len(map_uid)

    # frame counts for node size
    frame_counts = ensure_frame_counts(G, map_uid)

    # node colors and legend bar
    if node_custom_color is not None:
        node_colors = np.array(node_custom_color)
        cbar_label = node_custom_color_title
    else:
        node_colors = frame_counts
        cbar_label = "Frames (Population)"
        palette = 'viridis' # default is viridis
    
    # start node is just first frame and folded node is node with most counts
    start_node = int(map_uid[start_frame]) if 0 <= start_frame < F else int(map_uid[0])
    folded_node = int(np.argmax(frame_counts))

    # pos provided from an embedding in arguments but if not we can
    # extract them from the embedding if that's provided
    if pos is None:
        if X_emb is not None:
            X_emb = np.asarray(X_emb)
            pos = {i: (float(X_emb[i,0]), float(X_emb[i,1])) for i in range(U)}
        else: # if no pos or emb just normal networkx algorithm
            pos = nx.spring_layout(G, seed=42)
    
    node_x = np.array([pos[i][0] for i in range(U)])
    node_y = np.array([pos[i][1] for i in range(U)])

    # node sizes are frame counts
    min_size, max_size = node_size_range
    node_sizes = min_size + (frame_counts / max(1, frame_counts.max())) * (max_size - min_size)
    if 0 <= folded_node < U:
        node_sizes[folded_node] *= 1.35
    if 0 <= start_node < U:
        node_sizes[start_node] *= 2.0

    # Logic to split sequence into Pre and Post 
    try:
        first_unfold_frame = int(np.where(map_uid == start_node)[0][0])
    except IndexError:
        first_unfold_frame = 0
        
    later_idx = np.where(map_uid[first_unfold_frame:] == folded_node)[0]
    first_fold_frame = None if later_idx.size == 0 else int(first_unfold_frame + later_idx[0])

    # EDGES
    # -----
    # post folded node
    if first_fold_frame is None or first_fold_frame + 1 >= F:
        post_node_seq = []
    else:
        post_node_seq = compress_sequence(map_uid[first_fold_frame + 1 :])
    post_edges = expand_seq_to_edges(G, post_node_seq, expand_jumps, count_multiplicity=count_multiplicity)

    # pre folded node
    if first_fold_frame is None:
        pre_frames = map_uid[first_unfold_frame : ]
    else:
        pre_frames = map_uid[first_unfold_frame : first_fold_frame + 1]
    pre_node_seq = compress_sequence(pre_frames)
    pre_edges = expand_seq_to_edges(G, pre_node_seq,  expand_jumps, count_multiplicity=count_multiplicity)
    
    # shortest path (manifold and temporal)
    shortest_edge_pairs = []
    if show_shortest:
        try:
            shortest_nodes = nx.shortest_path(G, source=start_node, target=folded_node)
            for i in range(len(shortest_nodes)-1):
                a,b = int(shortest_nodes[i]), int(shortest_nodes[i+1])
                if a!=b: shortest_edge_pairs.append((a,b))
        except nx.NetworkXNoPath:
            pass

    # custom shortest paths (from args)
    paths = dict(zip(list(custom_paths.keys()), [[] for _ in range(len(custom_paths))]))
    for path in custom_paths:
        for i in range(len(custom_paths[path])-1):
            a, b = int(custom_paths[path][i]), int(custom_paths[path][i+1])
            if a != b: paths[path].append((a, b))

    # helpers to prepare the edge segments for matplotlib
    def pairs_to_segs(pairs):
        return [((pos[a][0], pos[a][1]), (pos[b][0], pos[b][1])) for a,b in pairs]

    #bg_segs = pairs_to_segs(list(G.edges()))
    temporal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('temporal', False)]
    bg_segs = pairs_to_segs(temporal_edges)

    post_segs, post_widths = pairs_and_widths_from_edges(post_edges, pos)
    pre_segs, pre_widths = pairs_and_widths_from_edges(pre_edges, pos)
    short_segs, short_widths = pairs_and_widths_from_edges(shortest_edge_pairs, pos)

    # Plotting
    # --------
    fig, ax = plt.subplots(figsize=figsize)

    # Background edges are light grey
    if bg_segs and show_bg:
        ax.add_collection(LineCollection(bg_segs, colors=[(200/255,200/255,200/255,0.5)], linewidths=1.0, zorder=1))

    # scatter nodes
    sc = ax.scatter(node_x, node_y, s=node_sizes**2, c=node_colors, cmap=palette, edgecolors='none', zorder=2)

    # node outlines
    for i in range(U):
        lw = 0.6
        ec = (80/255,80/255,80/255,0.35)
        if i == start_node: lw = 4.0; ec = (0,0,0,1) # special outline for start node (black)
        ax.scatter([node_x[i]], [node_y[i]], s=max(1, node_sizes[i]**2), facecolors='none', edgecolors=[ec], linewidths=lw, zorder=2.1)

    # draw pos-fold edges (red)
    if post_segs and post_fold_red:
        ax.add_collection(LineCollection(post_segs, colors=[(200/255,20/255,20/255,0.6)], linewidths=post_widths, zorder=3, linestyle='--'))

    # draw pre-fold background edges (grey)
    if pre_segs and show_bg:
        ax.add_collection(LineCollection(pre_segs, colors=[(150/255,150/255,150/255,0.4)], linewidths=pre_widths, zorder=4))

    # shortest path edges in blue
    if short_segs:
        ax.add_collection(LineCollection(short_segs, colors=[(0,0,200/255,1.0)], linewidths=3, zorder=6))
        sp_nodes_set = sorted({a for a,b in shortest_edge_pairs} | {b for a,b in shortest_edge_pairs})
        if sp_nodes_set:
            ax.scatter(node_x[sp_nodes_set], node_y[sp_nodes_set], s=64, c=[(0,0,200/255,1.0)], zorder=6.5)

    # any custom paths and their associated colors (args) or default color to biolet
    if len(paths) > 0:
        for i, path in enumerate(paths):
            segs, _ = pairs_and_widths_from_edges(paths[path], pos)
        
            if custom_paths_colors: # custom color
                custom_edge_color = custom_paths_colors[i]
            else: # default to violet
                if i == 0: custom_paths_colors = []
                custom_edge_color = (180/255, 20/255, 220/255, 1.0) 
                custom_paths_colors.append(custom_edge_color)
            ax.add_collection(LineCollection(segs, colors=[custom_edge_color], linewidths=3, zorder=7, label=path))
        
            # add markers for custom path nodes
            custom_nodes_set = sorted({a for a,b in paths[path]} | {b for a,b in paths[path]})
            if custom_nodes_set:
                ax.scatter(node_x[custom_nodes_set], node_y[custom_nodes_set], s=40, c=[custom_edge_color], zorder=7.1)

    # overlay on start node (black)
    if 0 <= start_node < U:
        ax.scatter([node_x[start_node]], [node_y[start_node]],
                   s=max(160, (node_sizes[start_node]*2.0)**2), c=[(0.0,0,0,0.6)], edgecolors='black', linewidths=2.5, zorder=8)

    # folded node with red star
    if folded_node:
        ax.scatter(node_x[folded_node], node_y[folded_node], linewidths=2, edgecolors='black',
           c='red', s=500, marker='*', label='True Folded State', zorder=10)

    # colorbar 
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)

    # debug text
    dbg = f"pre_pairs={len(pre_edges)} post_pairs={len(post_edges)} shortest_pairs={len(shortest_edge_pairs)} first_fold={first_fold_frame}"
    ax.text(0.01, -0.03, dbg, transform=ax.transAxes, fontsize=9, va='top')

    # legend
    custom_lines = []
    legend_labels = []
    if post_fold_red:
        custom_lines.append(Line2D([0], [0], color=(200/255,20/255,20/255,0.9), lw=3, linestyle='--'))
        legend_labels.append('Post-fold')
    if show_shortest: 
        custom_lines.append(Line2D([0], [0], color=(0,0,200/255), lw=4))
        legend_labels.append('Shortest Path')
    if len(paths) > 0:
        for i, path in enumerate(paths):
            custom_lines.append(Line2D([0], [0], color=custom_paths_colors[i], lw=3))
            legend_labels.append(path)
    ax.legend(custom_lines, legend_labels, loc='upper right')

    plt.tight_layout()
    return fig, ax

def plot_graph_widget():
    return
# TO DO NEEDS UPDATING WITH FUNCTIONALITY OF STATIC VERSION
# def plot_graph_widget(
#     G, 
#     pos, 
#     map_uid, 
#     start_frame=0, 
#     expand_jumps=True, 
#     show_shortest=True, 
#     physics_path_nodes=None,
#     node_custom_color=None,
#     node_custom_color_title="Value",
#     count_multiplicity=False,
#     unique_maps=None, 
#     unique_indices=None, 
#     figsize=(1100, 700), 
#     palette='RdBu_r'
# ):
#     '''
#     Plots an interactive plotly notebook widget graph given networkx graph object G. 
#     Nodes can be clicked to reveal a heatmap of their contact map on the right hand side.
#     Highlights post-folded state (most occupied node) edges red and the shortest
#     path edges blue. Node sizes correspond to counts in the trajectory and nodes are
#     coloured by their first frame's index in the trajectory.

#     pos can be provided to set node positions from say a clustering projection/embedding.
#     Or X_emb can be provided instead of pos (usually when generated by clustering).
#     '''
    
#     map_uid = np.asarray(map_uid, dtype=int)
#     U = G.number_of_nodes()

#     xs = np.array([pos[i][0] for i in range(U)])
#     ys = np.array([pos[i][1] for i in range(U)])

#     # 1. Size Calculation (Always based on Population/Stability)
#     frame_counts = ensure_frame_counts(G, map_uid)
#     start_node = int(map_uid[start_frame]) if 0 <= start_frame < len(map_uid) else int(map_uid[0])
#     folded_node = int(np.argmax(frame_counts)) if frame_counts.size else 0

#     # 2. Color Calculation
#     if node_custom_color is not None:
#         # Use provided physics values (e.g., Committor q)
#         node_colors = np.array(node_custom_color)
#         cbar_title = node_custom_color_title
#         hover_template = "node %{customdata}<br>frames: %{text}<br>" + f"{cbar_title}: %{{marker.color:.3f}}"
#     else:
#         # Default to frame counts
#         node_colors = frame_counts
#         cbar_title = "Frames"
#         hover_template = "node %{customdata}<br>frames: %{text}"

#     xs = np.array([pos[i][0] for i in range(U)])
#     ys = np.array([pos[i][1] for i in range(U)])

#     # Visual helpers
#     def node_visuals_plotly(counts, s_node, f_node):
#         denom = counts.max() if counts.max() > 0 else 1.0
#         sizes = 8 + (counts / denom) * 40
#         l_widths = np.full(len(sizes), 0.6)
#         l_colors = ['rgba(80,80,80,0.35)'] * len(sizes)
#         if 0 <= f_node < len(sizes):
#             l_widths[f_node] = 3.0; l_colors[f_node] = 'black'; sizes[f_node] *= 1.35
#         if 0 <= s_node < len(sizes):
#             l_widths[s_node] = 4.0; l_colors[s_node] = 'black'; sizes[s_node] *= 2.0
#         return sizes, l_widths, l_colors

#     def edge_traces_from_edges(edges):
#         traces = []
#         if isinstance(edges, Counter):
#             for (a,b), cnt in edges.items():
#                 xa, ya = pos[a]; xb, yb = pos[b]
#                 width = 2 + 1.5 * np.sqrt(cnt)
#                 hover = f"{a}-{b} count={cnt}"
#                 traces.append(dict(x=[xa, xb, None], y=[ya, yb, None], width=width, hover=hover))
#         else:
#             for a,b in edges:
#                 xa, ya = pos[a]; xb, yb = pos[b]
#                 traces.append(dict(x=[xa, xb, None], y=[ya, yb, None], width=3, hover=''))
#         return traces

#     node_sizes, line_widths, line_colors = node_visuals_plotly(frame_counts, start_node, folded_node)
#     hover_text = [f"node {i}<br>frames: {int(frame_counts[i])}" for i in range(U)]

#     # Helper for plotly segments
#     def edges_to_coords(pairs):
#         x, y = [], []
#         for a, b in pairs:
#             x += [pos[a][0], pos[b][0], None]
#             y += [pos[a][1], pos[b][1], None]
#         return x, y

#     # Edges
#     bg_pairs = list(G.edges())
#     bg_x, bg_y = edges_to_coords(bg_pairs)

#     # --- Pre/Post Logic (Replicated from Static for consistency) ---
#     try:
#         first_unfold_frame = int(np.where(map_uid == start_node)[0][0])
#     except IndexError:
#         first_unfold_frame = 0
#     later_idx = np.where(map_uid[first_unfold_frame:] == folded_node)[0]
#     first_fold_frame = None if later_idx.size == 0 else int(first_unfold_frame + later_idx[0])

#     if first_fold_frame is None or first_fold_frame + 1 >= len(map_uid):
#         post_seq = []
#     else:
#         post_seq = compress_sequence(map_uid[first_fold_frame+1:])
#     post_edges = expand_seq_to_edges(G, post_seq, expand_jumps, count_multiplicity=count_multiplicity)
#     #post_x, post_y = edge_traces_from_edges(post_pairs, pos, count_multiplicity=count_multiplicity)

#     if first_fold_frame is None:
#         pre_frames = map_uid[first_unfold_frame:]
#     else:
#         pre_frames = map_uid[first_unfold_frame:first_fold_frame+1]
#     pre_seq = compress_sequence(pre_frames)
#     pre_edges = expand_seq_to_edges(G, pre_seq, expand_jumps, count_multiplicity=count_multiplicity)
#     #pre_x, pre_y = edge_traces_from_edges(pre_pairs, pos, count_multiplicity=count_multiplicity)
#     # -------------------------------------------------------------

#     fig = make_subplots(rows=1, cols=2, column_widths=[0.66, 0.34], specs=[[{"type":"scatter"}, {"type":"heatmap"}]])

#     # 1. Background (Manifold)
#     if bg_x:
#         fig.add_trace(go.Scatter(x=bg_x, y=bg_y, mode='lines', line=dict(color='rgba(200,200,200,0.5)', width=1), hoverinfo='none'), row=1, col=1)

#     # 2. Post-fold (Red)
#     for t in edge_traces_from_edges(post_edges):
#         fig.add_trace(
#             go.Scatter(
#                 x=t['x'],
#                 y=t['y'],
#                 mode='lines+markers',
#                 line=dict(
#                     color='rgba(200,20,20,0.9)',
#                     width=t['width']
#                 ),
#                 hoverinfo='text',
#                 hovertext=[t['hover']],
#                 name='post'
#             ),
#             row=1, col=1
#         )
        
#     # 3. Pre-fold (Grey)
#     for t in edge_traces_from_edges(pre_edges):
#         fig.add_trace(
#             go.Scatter(
#                 x=t['x'],
#                 y=t['y'],
#                 mode='lines+markers',
#                 line=dict(
#                     color='rgba(150,150,150,0.6)',
#                     width=t['width']
#                 ),
#                 hoverinfo='text',
#                 hovertext=[t['hover']],
#                 name='post'
#             ),
#             row=1, col=1
#         )

#     # 4. Shortest (Blue)
#     if show_shortest:
#         try:
#             sp = nx.shortest_path(G, start_node, folded_node)
#             sp_pairs = [(sp[i], sp[i+1]) for i in range(len(sp)-1)]
#             sx, sy = edges_to_coords(sp_pairs)
#             if sx:
#                 fig.add_trace(go.Scatter(x=sx, y=sy, mode='lines+markers', line=dict(color='rgba(0,0,200,1.0)', width=4), marker=dict(size=8), name='shortest'), row=1, col=1)
#         except: pass

#     # 5. Physics Shortest Path (Violet)
#     if physics_path_nodes and len(physics_path_nodes) > 1:
#         pp_pairs = [(physics_path_nodes[i], physics_path_nodes[i+1]) for i in range(len(physics_path_nodes)-1)]
#         px, py = edges_to_coords(pp_pairs)
#         if px:
#              fig.add_trace(
#                  go.Scatter(x=px, y=py, mode='lines+markers', 
#                             line=dict(color='rgb(180, 20, 220)', width=3), # Violet
#                             marker=dict(size=6, color='rgb(180, 20, 220)'),
#                             name='Physics Path'), 
#                  row=1, col=1
#              )

#     # 6. Nodes (UPDATED FOR CUSTOM COLOR)
#     node_trace = go.Scatter(
#         x=xs, y=ys, mode='markers',
#         marker=dict(
#             size=node_sizes, 
#             color=node_colors, 
#             colorscale=palette, 
#             colorbar=dict(title=cbar_title),
#             showscale=True, 
#             line=dict(width=line_widths, color=line_colors)
#         ),
#         text=frame_counts, # Passed to hover template
#         customdata=np.arange(U), 
#         hovertemplate=hover_template,
#         name='nodes'
#     )
#     fig.add_trace(node_trace, row=1, col=1)

#     # 7. Heatmap
#     if unique_maps is not None:
#         fig.add_trace(go.Heatmap(z=unique_maps[start_node], zmin=0, zmax=1), row=1, col=2)
#     else:
#         fig.add_trace(go.Heatmap(z=np.zeros((2,2))), row=1, col=2)

#     fig.update_layout(height=figsize[1], width=figsize[0], title_text="Protein Folding Contact Topology Graph")
#     fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
#     fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)

#     figw = go.FigureWidget(fig)
#     info = HTML(value=f"Start: {start_node}, Folded: {folded_node}")

#     def on_click(trace, points, state):
#         if not points.point_inds: return
#         idx = points.point_inds[0]
#         node_id = trace.customdata[idx]
#         if unique_maps is not None:
#             figw.data[-1].z = unique_maps[node_id]
#         info.value = f"Node {node_id} (Frames: {G.nodes[node_id]['frame_count']})"

#     for tr in figw.data:
#         if tr.name == 'nodes': tr.on_click(on_click)

#     display(VBox([figw, info]))
#     return figw, info

def plot_graph_auto(
    G,
    map_uid,
    pos=None,
    X_emb=None,
    unique_maps=None,
    unique_indices=None,
    start_frame=0,
    expand_jumps=True,
    show_shortest=True,
    folded_node=None,
    show_bg=True,
    post_fold_red=True,
    custom_paths={},
    custom_paths_colors=None,
    node_custom_color=None,
    node_custom_color_title="custom",
    interactive=True,
    count_multiplicity=False,
    palette='Viridis',
    figsize_widget=(1100,700),
    figsize_static=(12,7),
    node_size_range=(8, 48),
    title=""
):
    '''  
    Master plotting wrapper so only one plotting function need be called. Simply specifying
    the interactive arg will determine whether a static or interactive plot is generated.
    '''
    map_uid = np.asarray(map_uid, dtype=int)
    
    # standardize layout (pos)
    if pos is None:
        if X_emb is not None:
            X_emb = np.asarray(X_emb)
            pos = {i: (float(X_emb[i,0]), float(X_emb[i,1])) for i in range(G.number_of_nodes())}
        else:
            try:
                pos = nx.spring_layout(G, seed=42)
            except:
                pos = {i:(0,0) for i in G.nodes()}

    # choose plotter
    if interactive:
        return plot_graph_widget(
            G, pos, map_uid,
            start_frame=start_frame,
            expand_jumps=expand_jumps,
            show_shortest=show_shortest,
            folded_node=folded_node,
            show_bg=show_bg,
            custom_paths=custom_paths,
            custom_paths_colors=custom_paths_colors,
            node_custom_color=node_custom_color,
            node_custom_color_title=node_custom_color_title,
            unique_maps=unique_maps,
            unique_indices=unique_indices,
            figsize=figsize_widget,
            palette=palette,
            count_multiplicity=count_multiplicity,
            title=title
        )
    else:
        # Pass all args specific to the old static logic
        return plot_graph_static(
            G, 
            map_uid=map_uid, 
            pos=pos,
            start_frame=start_frame,
            expand_jumps=expand_jumps,
            show_shortest=show_shortest,
            folded_node=folded_node,
            show_bg=show_bg,
            post_fold_red=post_fold_red,
            custom_paths=custom_paths,
            custom_paths_colors=custom_paths_colors,
            node_custom_color=node_custom_color,
            node_custom_color_title=node_custom_color_title,
            figsize=figsize_static,
            node_size_range=node_size_range,
            unique_maps=unique_maps,
            unique_indices=unique_indices,
            palette=palette,
            count_multiplicity=count_multiplicity,
            title=title
        )
    
def plot_energy_landscape(G, committor_map, free_energy_map, 
                         custom_paths=None, 
                         custom_paths_colors=None,
                         folded_node=None,
                         node_sizes=[],
                         legend_location="upper center",
                         figsize=(10, 6)):
    ''' 
    Alternative plotting coordinates for graph. Instead given a 
    mapping of commitor probability and free energy for each node 
    values for each node we plot them on the x,y respectively.
    '''
    
    nodes = list(G.nodes())
    x_q = []
    y_f = []
    if len(node_sizes) == 0: # default uniform node sizes of 40
        node_sizes = [40 for _ in range(len(nodes))]
    
    # check bounds for normalization
    f_values = [free_energy_map.get(n, 0) for n in nodes]
    min_f, max_f = min(f_values), max(f_values)
    
    for n in nodes: # for each node we get its respective q and f values = x,y coords
        q = committor_map.get(n, 0.0)
        f = free_energy_map.get(n, 0.0)
        
        # plus some jitter to x so nodes at exactly same q don't overlap perfectly
        jitter = np.random.normal(0, 0.005) 
        
        x_q.append(q + jitter)
        y_f.append(f)

    fig, ax = plt.subplots(figsize=figsize)
    
    # grey edges
    lines = []
    for u, v in G.edges():
        if u in committor_map and v in committor_map:
            xu, yu = committor_map[u], free_energy_map[u]
            xv, yv = committor_map[v], free_energy_map[v]
            lines.append([(xu, yu), (xv, yv)])
            
    lc = LineCollection(lines, colors='gray', alpha=0.15, linewidths=0.5, zorder=1)
    ax.add_collection(lc)

    # draw nodes coloured by node sizes
    sc = ax.scatter(x_q, y_f, c=x_q, cmap='RdBu_r', s=node_sizes, 
                    edgecolor='k', linewidth=0.3, alpha=0.8, zorder=2)

    # draw custom paths
    if custom_paths:
        for i, path in enumerate(custom_paths):
            # Draw the line
            path_x = [committor_map[n] for n in custom_paths[path]]
            path_y = [free_energy_map[n] for n in custom_paths[path]]
            if custom_paths_colors:
                path_color = custom_paths_colors[i]
            else: path_color = "violet"
            ax.plot(path_x, path_y, color=path_color, linewidth=1, label=path, zorder=3)
            ax.scatter(path_x, path_y, color=path_color, s=20, zorder=4)

    if folded_node:
        ax.scatter(x_q[folded_node], y_f[folded_node], linewidths=2, edgecolors='black',
            c='red', s=500, marker='*', label='True Folded State', zorder=10)

    # formatting
    ax.set_xlabel("Reaction Coordinate $q$ (Committor Probability)")
    ax.set_ylabel("Free Energy $F$ ($k_B T$)")
    ax.set_title("Chignolin Projected Free Energy Landscape on Reaction Coordinates")
    
    # state text (this is for my specific results when I ran the code, too lazy to make as function arguments!)
    ax.text(0.05, 0.25, "Unfolded Basin", transform=ax.transAxes, ha='left', va='top', fontweight='bold', color='blue')
    ax.text(0.9, 0.1, "Folded Basin", transform=ax.transAxes, ha='right', va='top', fontweight='bold', color='red')
    ax.text(0.7, 0.65, "Transition Barrier", transform=ax.transAxes, ha='center', va='bottom', fontweight='bold', alpha=0.5)

    cbar = plt.colorbar(sc)
    cbar.set_label("Folding Prob ($q$)")
    ax.legend(loc=legend_location)
    
    return fig, ax