import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import HTML, VBox
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import Normalize


def ensure_frame_counts(G, map_uid):
    """Ensure G.nodes[n]['frame_count'] exists."""
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
    """Compress consecutive duplicates, preserve order."""
    seq = list(seq)
    if not seq:
        return []
    out = [int(seq[0])]
    for x in seq[1:]:
        xi = int(x)
        if xi != out[-1]:
            out.append(xi)
    return out

def expand_seq_to_edges(G, node_seq, expand_jumps=True):
    """
    Convert compressed node sequence to undirected edge pairs. 
    If expand_jumps=True, fills gaps using shortest_path.
    """
    seen = set()
    out = []
    for i in range(len(node_seq)-1):
        a = int(node_seq[i]); b = int(node_seq[i+1])
        if a == b:
            continue
        # If edge exists physically
        if G.has_edge(a,b) or G.has_edge(b,a):
            key = tuple(sorted((a,b)))
            if key not in seen:
                seen.add(key); out.append((a,b))
            continue
        # If jump needs expansion
        if expand_jumps:
            try:
                path_nodes = nx.shortest_path(G, source=a, target=b)
            except nx.NetworkXNoPath:
                continue
            for j in range(len(path_nodes)-1):
                u = int(path_nodes[j]); v = int(path_nodes[j+1])
                key = tuple(sorted((u,v)))
                if key not in seen:
                    seen.add(key); out.append((u,v))
    return out

def plot_graph_static(
    G,
    map_uid,
    pos=None,
    X_emb=None,
    start_frame=0,
    expand_jumps=True,
    show_shortest=True,
    figsize=(12, 7),
    node_size_range=(8, 48),
    unique_maps=None,     # Ignored in static, kept for signature compatibility
    unique_indices=None,  # Ignored in static, kept for signature compatibility
    title="Contact-difference graph (Static)",
):
    """
    Exact implementation of the 'Old Function' logic within the new framework.
    Does NOT rely on edge attributes. Calculates Pre/Post/Shortest paths from map_uid.
    """
    if map_uid is None:
        raise RuntimeError("map_uid required for static plot")
    
    map_uid = np.asarray(map_uid, dtype=int)
    U = G.number_of_nodes()
    F = len(map_uid)

    # 1. Ensure frame counts
    frame_counts = ensure_frame_counts(G, map_uid)

    # 2. Determine Start and Folded Nodes
    start_node = int(map_uid[start_frame]) if 0 <= start_frame < F else int(map_uid[0])
    folded_node = int(np.argmax(frame_counts))

    # 3. Handle Positions (Priority: pos -> X_emb -> spring)
    if pos is None:
        if X_emb is not None:
            X_emb = np.asarray(X_emb)
            pos = {i: (float(X_emb[i,0]), float(X_emb[i,1])) for i in range(U)}
        else:
            pos = nx.spring_layout(G, seed=42)
    
    node_x = np.array([pos[i][0] for i in range(U)])
    node_y = np.array([pos[i][1] for i in range(U)])

    # 4. Node Sizes (Old aesthetic logic)
    min_size, max_size = node_size_range
    node_sizes = min_size + (frame_counts / max(1, frame_counts.max())) * (max_size - min_size)
    if 0 <= folded_node < U:
        node_sizes[folded_node] *= 1.35
    if 0 <= start_node < U:
        node_sizes[start_node] *= 2.0

    # 5. Logic to split sequence into Pre and Post (Exact old logic)
    try:
        first_unfold_frame = int(np.where(map_uid == start_node)[0][0])
    except IndexError:
        first_unfold_frame = 0
        
    later_idx = np.where(map_uid[first_unfold_frame:] == folded_node)[0]
    first_fold_frame = None if later_idx.size == 0 else int(first_unfold_frame + later_idx[0])

    # -- Post Sequence --
    if first_fold_frame is None or first_fold_frame + 1 >= F:
        post_node_seq = []
    else:
        post_node_seq = compress_sequence(map_uid[first_fold_frame + 1 :])
    post_edge_pairs = expand_seq_to_edges(G, post_node_seq, expand_jumps)

    # -- Pre Sequence --
    if first_fold_frame is None:
        pre_frames = map_uid[first_unfold_frame : ]
    else:
        pre_frames = map_uid[first_unfold_frame : first_fold_frame + 1]
    pre_node_seq = compress_sequence(pre_frames)
    pre_edge_pairs = expand_seq_to_edges(G, pre_node_seq, expand_jumps)

    # -- Shortest Path --
    shortest_edge_pairs = []
    if show_shortest:
        try:
            shortest_nodes = nx.shortest_path(G, source=start_node, target=folded_node)
            for i in range(len(shortest_nodes)-1):
                a,b = int(shortest_nodes[i]), int(shortest_nodes[i+1])
                if a!=b: shortest_edge_pairs.append((a,b))
        except nx.NetworkXNoPath:
            pass

    # 6. Prepare Segments for Matplotlib
    def pairs_to_segs(pairs):
        return [((pos[a][0], pos[a][1]), (pos[b][0], pos[b][1])) for a,b in pairs]

    bg_segs = pairs_to_segs(list(G.edges()))
    post_segs = pairs_to_segs(post_edge_pairs)
    pre_segs = pairs_to_segs(pre_edge_pairs)
    short_segs = pairs_to_segs(shortest_edge_pairs)

    # 7. Plotting (Exact Matplotlib commands from old function)
    fig, ax = plt.subplots(figsize=figsize)

    # Background (light grey)
    if bg_segs:
        ax.add_collection(LineCollection(bg_segs, colors=[(200/255,200/255,200/255,0.5)], linewidths=1.0, zorder=1))

    # Nodes
    sc = ax.scatter(node_x, node_y, s=node_sizes**2, c=frame_counts, cmap='viridis', edgecolors='none', zorder=2)

    # Node Outlines
    # Vectorized outline drawing for speed, specific highlights for start/fold
    # (Reverting to loop to match old logic exactly allows individual control)
    for i in range(U):
        lw = 0.6
        ec = (80/255,80/255,80/255,0.35)
        if i == start_node: lw = 4.0; ec = (0,0,0,1)
        # Note: Old function had commented out folded_node outline, keeping that logic
        ax.scatter([node_x[i]], [node_y[i]], s=max(1, node_sizes[i]**2), facecolors='none', edgecolors=[ec], linewidths=lw, zorder=2.1)

    # Post-fold edges (Red) - Draw BEFORE pre
    if post_segs:
        ax.add_collection(LineCollection(post_segs, colors=[(200/255,20/255,20/255,0.9)], linewidths=3, zorder=3))
        # Markers for post nodes
        post_nodes_set = sorted({a for a,b in post_edge_pairs} | {b for a,b in post_edge_pairs})
        if post_nodes_set:
            ax.scatter(node_x[post_nodes_set], node_y[post_nodes_set], s=36, c=[(200/255,20/255,20/255,0.9)], zorder=3.5)

    # Pre-fold edges (Translucent Grey)
    if pre_segs:
        ax.add_collection(LineCollection(pre_segs, colors=[(150/255,150/255,150/255,0.4)], linewidths=2, zorder=4))

    # Shortest path (Blue)
    if short_segs:
        ax.add_collection(LineCollection(short_segs, colors=[(0,0,200/255,1.0)], linewidths=3, zorder=6))
        sp_nodes_set = sorted({a for a,b in shortest_edge_pairs} | {b for a,b in shortest_edge_pairs})
        if sp_nodes_set:
            ax.scatter(node_x[sp_nodes_set], node_y[sp_nodes_set], s=64, c=[(0,0,200/255,1.0)], zorder=6.5)

    # Start Node Top Layer
    if 0 <= start_node < U:
        ax.scatter([node_x[start_node]], [node_y[start_node]],
                   s=max(160, (node_sizes[start_node]*2.0)**2), c=[(0.0,0,0,0.6)], edgecolors='black', linewidths=2.5, zorder=8)

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('frames')

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)

    # Debug Text
    dbg = f"pre_pairs={len(pre_edge_pairs)} post_pairs={len(post_edge_pairs)} shortest_pairs={len(shortest_edge_pairs)} first_fold={first_fold_frame}"
    ax.text(0.01, -0.03, dbg, transform=ax.transAxes, fontsize=9, va='top')

    plt.tight_layout()
    return fig, ax

def plot_graph_widget(
    G, pos, map_uid, 
    start_frame=0, expand_jumps=True, show_shortest=True, 
    unique_maps=None, unique_indices=None, figsize=(1100, 700), palette='Viridis'
):
    """Interactive Plotly widget."""
    
    map_uid = np.asarray(map_uid, dtype=int)
    U = G.number_of_nodes()
    frame_counts = ensure_frame_counts(G, map_uid)

    start_node = int(map_uid[start_frame]) if 0 <= start_frame < len(map_uid) else int(map_uid[0])
    folded_node = int(np.argmax(frame_counts)) if frame_counts.size else 0

    xs = np.array([pos[i][0] for i in range(U)])
    ys = np.array([pos[i][1] for i in range(U)])

    # Visual helpers
    def node_visuals_plotly(counts, s_node, f_node):
        denom = counts.max() if counts.max() > 0 else 1.0
        sizes = 8 + (counts / denom) * 40
        l_widths = np.full(len(sizes), 0.6)
        l_colors = ['rgba(80,80,80,0.35)'] * len(sizes)
        if 0 <= f_node < len(sizes):
            l_widths[f_node] = 3.0; l_colors[f_node] = 'black'; sizes[f_node] *= 1.35
        if 0 <= s_node < len(sizes):
            l_widths[s_node] = 4.0; l_colors[s_node] = 'black'; sizes[s_node] *= 2.0
        return sizes, l_widths, l_colors

    node_sizes, line_widths, line_colors = node_visuals_plotly(frame_counts, start_node, folded_node)
    
    hover_text = [f"node {i}<br>frames: {int(frame_counts[i])}" for i in range(U)]

    # Helper for plotly segments
    def edges_to_coords(pairs):
        x, y = [], []
        for a, b in pairs:
            x += [pos[a][0], pos[b][0], None]
            y += [pos[a][1], pos[b][1], None]
        return x, y

    # Edges
    bg_pairs = list(G.edges())
    bg_x, bg_y = edges_to_coords(bg_pairs)

    # --- Pre/Post Logic (Replicated from Static for consistency) ---
    try:
        first_unfold_frame = int(np.where(map_uid == start_node)[0][0])
    except IndexError:
        first_unfold_frame = 0
    later_idx = np.where(map_uid[first_unfold_frame:] == folded_node)[0]
    first_fold_frame = None if later_idx.size == 0 else int(first_unfold_frame + later_idx[0])

    if first_fold_frame is None or first_fold_frame + 1 >= len(map_uid):
        post_seq = []
    else:
        post_seq = compress_sequence(map_uid[first_fold_frame+1:])
    post_pairs = expand_seq_to_edges(G, post_seq, expand_jumps)
    post_x, post_y = edges_to_coords(post_pairs)

    if first_fold_frame is None:
        pre_frames = map_uid[first_unfold_frame:]
    else:
        pre_frames = map_uid[first_unfold_frame:first_fold_frame+1]
    pre_seq = compress_sequence(pre_frames)
    pre_pairs = expand_seq_to_edges(G, pre_seq, expand_jumps)
    pre_x, pre_y = edges_to_coords(pre_pairs)
    # -------------------------------------------------------------

    fig = make_subplots(rows=1, cols=2, column_widths=[0.66, 0.34], specs=[[{"type":"scatter"}, {"type":"heatmap"}]])

    # 1. Background (Manifold)
    if bg_x:
        fig.add_trace(go.Scatter(x=bg_x, y=bg_y, mode='lines', line=dict(color='rgba(200,200,200,0.5)', width=1), hoverinfo='none'), row=1, col=1)

    # 2. Post-fold (Red)
    if post_x:
        fig.add_trace(go.Scatter(x=post_x, y=post_y, mode='lines+markers', line=dict(color='rgba(200,20,20,0.9)', width=3), marker=dict(size=6, color='rgba(200,20,20,0.9)'), name='post'), row=1, col=1)

    # 3. Pre-fold (Grey)
    if pre_x:
        fig.add_trace(go.Scatter(x=pre_x, y=pre_y, mode='lines', line=dict(color='rgba(150,150,150,0.6)', width=2), name='pre'), row=1, col=1)

    # 4. Shortest (Blue)
    if show_shortest:
        try:
            sp = nx.shortest_path(G, start_node, folded_node)
            sp_pairs = [(sp[i], sp[i+1]) for i in range(len(sp)-1)]
            sx, sy = edges_to_coords(sp_pairs)
            if sx:
                fig.add_trace(go.Scatter(x=sx, y=sy, mode='lines+markers', line=dict(color='rgba(0,0,200,1.0)', width=4), marker=dict(size=8), name='shortest'), row=1, col=1)
        except: pass

    # 5. Nodes
    node_trace = go.Scatter(
        x=xs, y=ys, mode='markers',
        marker=dict(size=node_sizes, color=frame_counts, colorscale=palette, showscale=True, line=dict(width=line_widths, color=line_colors)),
        hovertext=hover_text, hoverinfo='text', customdata=np.arange(U), name='nodes'
    )
    fig.add_trace(node_trace, row=1, col=1)

    # 6. Heatmap
    if unique_maps is not None:
        fig.add_trace(go.Heatmap(z=unique_maps[start_node], zmin=0, zmax=1), row=1, col=2)
    else:
        fig.add_trace(go.Heatmap(z=np.zeros((2,2))), row=1, col=2)

    fig.update_layout(height=figsize[1], width=figsize[0], title_text="Contact-difference graph (Interactive)")
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)

    figw = go.FigureWidget(fig)
    info = HTML(value=f"Start: {start_node}, Folded: {folded_node}")

    def on_click(trace, points, state):
        if not points.point_inds: return
        idx = points.point_inds[0]
        node_id = trace.customdata[idx]
        if unique_maps is not None:
            figw.data[-1].z = unique_maps[node_id]
        info.value = f"Node {node_id} (Frames: {G.nodes[node_id]['frame_count']})"

    for tr in figw.data:
        if tr.name == 'nodes': tr.on_click(on_click)

    display(VBox([figw, info]))
    return figw, info

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
    interactive=True,
    palette='Viridis',
    figsize_widget=(1100,700),
    figsize_static=(12,7),
    node_size_range=(8, 48)
):
    """
    Unified wrapper. 
    - If interactive=True: Returns (FigureWidget, HTML)
    - If interactive=False: Returns (fig, ax) using EXACT old aesthetic/logic.
    """
    map_uid = np.asarray(map_uid, dtype=int)
    
    # 1. Standardize Layout (Pos)
    if pos is None:
        if X_emb is not None:
            X_emb = np.asarray(X_emb)
            pos = {i: (float(X_emb[i,0]), float(X_emb[i,1])) for i in range(G.number_of_nodes())}
        else:
            try:
                pos = nx.spring_layout(G, seed=42)
            except:
                pos = {i:(0,0) for i in G.nodes()}

    # 2. Route to appropriate plotter
    if interactive:
        return plot_graph_widget(
            G, pos, map_uid,
            start_frame=start_frame,
            expand_jumps=expand_jumps,
            show_shortest=show_shortest,
            unique_maps=unique_maps,
            unique_indices=unique_indices,
            figsize=figsize_widget,
            palette=palette
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
            figsize=figsize_static,
            node_size_range=node_size_range,
            unique_maps=unique_maps,
            unique_indices=unique_indices
        )