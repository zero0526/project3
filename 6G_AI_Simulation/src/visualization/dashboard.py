import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import networkx as nx
import os
import time

# --- CONFIG ---
st.set_page_config(page_title="6G Digital Twin", layout="wide", page_icon="📡")
DATA_DIR = "data/"
STATE_FILE = os.path.join(DATA_DIR, "live_state.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history.csv")

# --- CSS CUSTOMIZATION ---
st.markdown("""
    <style>
        .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;}
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None

def draw_network(state):
    """Vẽ Topology mạng dùng NetworkX + Plotly"""
    G = nx.Graph()
    
    # Add Nodes
    for n in state['nodes']:
        G.add_node(n['id'], pos=(0,0), **n) # Pos sẽ được tính lại
    
    # Add Edges
    for l in state['links']:
        G.add_edge(l['source'], l['target'])
    
    # Tính Layout (Vị trí các node)
    # Dùng seed cố định để mạng không bị nhảy lung tung mỗi lần refresh
    pos = nx.spring_layout(G, seed=42) 
    
    # 1. Vẽ Edges (Lines)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    # 2. Vẽ Nodes (Markers)
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node_id in G.nodes():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node_id]
        util = node_data.get('cpu_util', 0)
        
        # Color scale: Green -> Yellow -> Red
        # Logic: 0.0 -> Green, 1.0 -> Red
        # Plotly colorscale handles numerical values automatically
        node_color.append(util)
        
        # Cloud node to hơn Edge node
        size = 30 if node_data.get('type') == 'cloud' else 20
        node_size.append(size)
        
        # Hover info
        svcs = node_data.get('active_services', [])
        info = f"<b>{node_id}</b><br>Type: {node_data.get('type')}<br>CPU: {util*100:.1f}%<br>Svcs: {svcs}"
        node_text.append(info)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n for n in G.nodes()], # Hiển thị ID trên node
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='RdYlGn_r', # Red-Yellow-Green (Reversed: Low=Green, High=Red)
            reversescale=False,
            color=node_color,
            size=node_size,
            cmin=0, cmax=1,
            colorbar=dict(
                thickness=15,
                title='CPU Load',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Network Topology & Node Health',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig

def draw_heatmap(state):
    """Vẽ Heatmap phân bố dịch vụ (Hàng: Node, Cột: Service ID)"""
    nodes = sorted(state['nodes'], key=lambda x: x['id'])
    node_ids = [n['id'] for n in nodes]
    
    # Xác định tổng số dịch vụ (giả sử max ID tìm thấy + 1 hoặc fix cứng 5)
    max_svc_id = 4 
    
    # Tạo ma trận Z (Binary)
    z_data = []
    for n in nodes:
        row = [0] * (max_svc_id + 1)
        for sid in n['active_services']:
            if sid <= max_svc_id:
                row[sid] = 1
        z_data.append(row)
        
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[f"Svc {i}" for i in range(max_svc_id + 1)],
        y=node_ids,
        colorscale=[[0, '#f0f2f6'], [1, '#004c6d']], # Trắng -> Xanh đậm
        showscale=False,
        xgap=2, ygap=2
    ))
    fig.update_layout(title="Service Placement Map", height=350, margin=dict(l=0, r=0, t=30, b=0))
    return fig

# --- MAIN APP ---
st.title("📡 6G AI Orchestration Digital Twin")

# Sidebar controls
st.sidebar.header("Controls")
refresh_rate = st.sidebar.slider("Refresh Rate (s)", 0.5, 5.0, 1.0)
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

# Load Data
state = load_state()

if state:
    # 1. Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Time Slot", state['step'])
    col2.metric("Total Energy (J)", f"{state['global_energy']:.2f}")
    col3.metric("QoS Violations", state['qos_violations'])
    col4.metric("Tasks Completed", state['completed_tasks'])

    # 2. Main Visuals
    c_left, c_right = st.columns([1.5, 1])
    
    with c_left:
        st.plotly_chart(draw_network(state), use_container_width=True)
        
    with c_right:
        st.plotly_chart(draw_heatmap(state), use_container_width=True)

    # 3. Historical Charts
    st.subheader("📊 Real-time Performance Metrics")
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        
        # Lấy 100 điểm gần nhất để đồ thị chạy
        df_view = df.tail(100)
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=df_view['step'], y=df_view['total_energy'], fill='tozeroy', name='Energy'))
            fig_e.update_layout(title="Energy Consumption (J)", height=250, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_e, use_container_width=True)
            
        with chart_col2:
            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(x=df_view['step'], y=df_view['qos_violations'], line=dict(color='red'), name='QoS'))
            fig_q.update_layout(title="Cumulative QoS Violations", height=250, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_q, use_container_width=True)

else:
    st.info("Waiting for simulation to start... Please run 'python main.py'")

# Logic Auto Refresh
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()