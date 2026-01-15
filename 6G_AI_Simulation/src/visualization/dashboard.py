import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import networkx as nx
import os
import time

# --- CONFIG ---
st.set_page_config(page_title="6G Simulation Dashboard", layout="wide", page_icon="📊")
DATA_DIR = "data/"
STATE_FILE = os.path.join(DATA_DIR, "live_state.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history.csv")

# --- CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #00f2fe; }
    </style>
""", unsafe_allow_html=True)

def load_state():
    if not os.path.exists(STATE_FILE): return None
    try:
        with open(STATE_FILE, "r") as f: return json.load(f)
    except: return None

def draw_network(state):
    G = nx.Graph()
    for n in state['nodes']: G.add_node(n['id'], **n)
    for l in state['links']: G.add_edge(l['source'], l['target'])
    
    pos = nx.spring_layout(G, seed=42)
    
    # Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#444'), mode='lines')

    # Nodes
    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    for node_id in G.nodes():
        x, y = pos[node_id]
        node_x.append(x); node_y.append(y)
        node_data = G.nodes[node_id]
        
        # MÀU SẮC THEO QUEUE (LOAD FACTOR)
        # 0.0 (Trống) -> Green, 0.5 -> Yellow, 1.0 (Quá tải) -> Red
        node_color.append(node_data.get('cpu_util', 0))
        node_size.append(35 if node_data.get('type') == 'cloud' else 25)
        
        info = f"Node: {node_id}<br>Type: {node_data['type']}<br>Queue: {node_data['backlog']:.1f} GFLOPS<br>Services: {node_data['active_services']}"
        node_text.append(info)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=[n for n in G.nodes()],
        textposition="bottom center", hoverinfo='text', hovertext=node_text,
        marker=dict(
            showscale=True, colorscale='RdYlGn_r', color=node_color,
            size=node_size, line_width=2, cmin=0, cmax=1,
            colorbar=dict(thickness=15, title=dict(text='Queue Load', side='right'), xanchor='left')
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    template="plotly_dark", height=600))
    return fig

# --- UI ---
st.title("🌐 6G Digital Twin - Realtime Monitoring")

state = load_state()

if not state:
    st.warning("⚠️ Đang chờ dữ liệu từ Simulation... Hãy chạy `python main.py` trước.")
    st.stop()

# Row 1: Metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Timeslot", state['step'])
m2.metric("Năng lượng (Slot)", f"{state['global_energy']:.4f} J")
m3.metric("Vi phạm QoS", state['qos_violations'])
m4.metric("Tasks Completed", state['completed_tasks'])
m5.metric("Incoming Tasks", state.get('incoming_tasks', 0))

# Row 2: Graph and Stats
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📍 Topology & Queue Status")
    st.plotly_chart(draw_network(state), use_container_width=True)

with c2:
    st.subheader("📋 Top Overloaded Nodes")
    df_nodes = pd.DataFrame(state['nodes'])
    df_overload = df_nodes[['id', 'type', 'backlog', 'active_services']].sort_values(by='backlog', ascending=False)
    
    # Style cho bảng
    st.dataframe(df_overload, use_container_width=True, height=400)
    
    st.subheader("📡 Service Installation")
    # Vẽ biểu đồ ngang cho mỗi node cài bao nhiêu service
    st.bar_chart(df_nodes.set_index('id')['active_services'].apply(len))

# Row 3: History
st.subheader("📈 Performance History")
if os.path.exists(HISTORY_FILE):
    df_hist = pd.read_csv(HISTORY_FILE).tail(50)
    st.line_chart(df_hist.set_index('step')[['total_energy', 'qos_violations']])

# Control
if st.sidebar.checkbox("Auto Refresh", True):
    time.sleep(1)
    st.rerun()