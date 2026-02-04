import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os
import json
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="6G Intelligent IOC",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Dark Theme CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00d4ff; }
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #2d3139;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .status-card {
        background: linear-gradient(135deg, #1a1c24 0%, #111319 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #00d4ff;
        box-shadow: 0 10px 15px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }
    .status-card h3 { color: #00d4ff; margin-bottom: 15px; }
    .status-card p { margin: 8px 0; font-size: 15px; }
    </style>
    """, unsafe_allow_html=True)

# Shared State for Topology Layout
if 'topo_pos' not in st.session_state:
    st.session_state.topo_pos = None

# Paths
history_path = "data/training_history.csv"
live_path = "data/live_state.json"

# Data Loading Functions
def get_history():
    if os.path.exists(history_path):
        try: return pd.read_csv(history_path)
        except: return None
    return None

def get_live():
    if os.path.exists(live_path):
        try:
            with open(live_path, 'r') as f: return json.load(f)
        except: return None
    return None

# Topology Visualization Logic
def draw_network_topology(state):
    if not state or 'nodes' not in state:
        return None
    
    nodes = state['nodes']
    links = state['links']
    
    G = nx.Graph()
    for n in nodes: G.add_node(n['id'])
    for l in links: G.add_edge(l['source'], l['target'])

    # Fix layout coordinates to avoid jumping
    if st.session_state.topo_pos is None or len(st.session_state.topo_pos) != len(G.nodes()):
        st.session_state.topo_pos = nx.spring_layout(G, seed=42, k=0.8)
    
    pos = st.session_state.topo_pos
    
    # Edge Visualization
    ex, ey = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            p1, p2 = pos[u], pos[v]
            ex.extend([p1[0], p2[0], None])
            ey.extend([p1[1], p2[1], None])
    
    edge_trace = go.Scatter(x=ex, y=ey, line=dict(width=1, color='#3d444d'), hoverinfo='none', mode='lines')

    # Node Visualization by Type
    type_meta = {
        'cloud': {'symbol': 'diamond', 'size': 20, 'is_active': True},
        'edge': {'symbol': 'circle', 'size': 16, 'is_active': True},
        'network': {'symbol': 'hexagon', 'size': 14, 'is_active': True},
        'terminal': {'symbol': 'square', 'size': 10, 'is_active': False},
        'relay': {'symbol': 'triangle-up', 'size': 12, 'is_active': False}
    }

    node_traces = []
    for t_name, meta in type_meta.items():
        subset = [n for n in nodes if n['type'] == t_name]
        if not subset: continue
        
        coords = [pos[n['id']] for n in subset if n['id'] in pos]
        if not coords: continue

        if meta['is_active']:
            colors = [n.get('energy', 0.0) for n in subset]
            cscale = 'YlOrRd'
            show_cb = (t_name == 'edge')
        else:
            colors = '#ffffff'
            cscale = None
            show_cb = False

        # Prepare hover strings
        hover_texts = []
        for n in subset:
            services = ", ".join(map(str, n.get('active_services', []))) if n.get('active_services') else "None"
            util = n.get('cpu_util', 0) * 100
            txt = (f"<b>Node: {n['id']}</b><br>"
                   f"Type: {t_name.upper()}<br>"
                   f"Capacity: {n.get('capacity', 0):.1f} GFLOPS<br>"
                   f"CPU Util: {util:.1f}%<br>"
                   f"Services: {services}<br>"
                   f"Energy: {n.get('energy',0):.4f}J<br>"
                   f"Backlog: {n.get('backlog',0):.1f}")
            hover_texts.append(txt)

        trace = go.Scatter(
            x=[c[0] for c in coords], y=[c[1] for c in coords],
            mode='markers', name=t_name.upper(),
            text=hover_texts,
            hoverinfo='text',
            marker=dict(
                symbol=meta['symbol'], size=meta['size'],
                color=colors, colorscale=cscale, showscale=show_cb,
                colorbar=dict(thickness=15, title="Energy (J)", x=1.05) if show_cb else None,
                line=dict(width=1.5, color='#111')
            )
        )
        node_traces.append(trace)

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        template="plotly_dark",
        margin=dict(b=0, l=0, r=0, t=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

# --- STREAMLIT UI LAYOUT ---
st.title("6G Network Intelligent Operations Center")
st.markdown("Automated Intelligent Management & Visualization System")

# Refresh Control in Sidebar
st.sidebar.title("🛠 IOC Control")
refresh_rate = st.sidebar.slider("Update Frequency (s)", 0.5, 5.0, 1.0)

history_data = get_history()
current_live = get_live()

# Row 1: High-Level Metrics
metrics_cols = st.columns(4)
if history_data is not None and not history_data.empty:
    curr_ep = history_data.iloc[-1]
    prev_ep = history_data.iloc[-2] if len(history_data) > 1 else curr_ep
    metrics_cols[0].metric("Reward", f"{curr_ep['reward']:.2f}", f"{curr_ep['reward'] - prev_ep['reward']:.2f}")
    metrics_cols[1].metric("Total Energy", f"{curr_ep['energy']:.1f} J")
    metrics_cols[2].metric("QoS Violation", f"{curr_ep['violation_rate']:.2f}%", f"{prev_ep['violation_rate'] - curr_ep['violation_rate']:.2f}%", delta_color="inverse")
    metrics_cols[3].metric("Q-Loss", f"{curr_ep['q_loss']:.2e}")

st.divider()

# Row 2: Topology and Node Inspection
col_map, col_info = st.columns([2, 1])

with col_map:
    st.subheader("Network(Atlanta) Energy Heatmap")
    if current_live:
        fig_topo = draw_network_topology(current_live)
        if fig_topo:
            st.plotly_chart(fig_topo, use_container_width=True)
    else:
        st.info("Loading live network state...")

with col_info:
    st.subheader("Node Diagnostics")
    if current_live:
        available_ids = sorted([n['id'] for n in current_live['nodes']])
        selected_id = st.selectbox("Select Node to Inspect:", available_ids)
        
        target_node = next((n for n in current_live['nodes'] if n['id'] == selected_id), None)
        if target_node:
            # RENDER STATUS CARD (CLEAN)
            st.markdown(f"""
            <div class="status-card">
                <h3>Node: {selected_id}</h3>
                <p><b>Equipment Type:</b> {target_node['type'].upper()}</p>
                <p><b>Computation Power:</b> {target_node.get('capacity', 0):.0f} GFLOPS</p>
                <p><b>Power Consumption:</b> {target_node.get('energy', 0):.6f} J</p>
                <p><b>Current Backlog:</b> {target_node.get('backlog', 0):.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Historical load chart for the selected node
            node_hist = target_node.get('history', [])
            if node_hist:
                fig_node = px.line(y=node_hist, template="plotly_dark", height=200)
                fig_node.update_layout(
                    margin=dict(l=0,r=0,t=10,b=0),
                    xaxis_visible=False,
                    yaxis_title="Backlog Load",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_node, use_container_width=True)

# Row 3: Overall Training Trends
st.divider()
st.subheader("Training Performance History")
if history_data is not None and not history_data.empty:
    chart_cols = st.columns(3)
    with chart_cols[0]:
        st.plotly_chart(px.line(history_data, x="episode", y="reward", title="Reward Growth", template="plotly_dark", color_discrete_sequence=['#00d4ff']), use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(px.line(history_data, x="episode", y="violation_rate", title="SLA Violation Rate (%)", template="plotly_dark", color_discrete_sequence=['#ff4b4b']), use_container_width=True)
    with chart_cols[2]:
        st.plotly_chart(px.line(history_data, x="episode", y="energy", title="Energy Consumption Trend", template="plotly_dark", color_discrete_sequence=['#ffaa00']), use_container_width=True)

# Auto-Refresh Control
time.sleep(refresh_rate)
st.rerun()
