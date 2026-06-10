import dash
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BASE_DIR
from src.models.bridge import Bridge

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.title = "桥梁结构健康监测系统"


def get_bridge_options():
    bridges = Bridge.list_all()
    return [{"label": b.name, "value": b.id} for b in bridges]


sidebar = dbc.Nav(
    [
        dbc.NavLink("首页", href="/", active="exact"),
        dbc.NavLink("数据导入", href="/data-import", active="exact"),
        dbc.NavLink("模态分析", href="/modal-analysis", active="exact"),
        dbc.NavLink("损伤检测", href="/damage-detection", active="exact"),
        dbc.NavLink("长期监控", href="/monitoring", active="exact"),
        dbc.NavLink("系统配置", href="/configuration", active="exact"),
        dbc.NavLink("报告生成", href="/report", active="exact"),
    ],
    vertical=True,
    pills=True,
    className="bg-light p-3",
    style={"height": "100vh", "width": "200px"},
)

app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="current-bridge-store", storage_type="session", data=None),
        dcc.Store(id="current-event-store", storage_type="session"),
        dcc.Store(id="baseline-store", storage_type="session"),
        dcc.Store(id="bridge-selector-trigger", storage_type="memory", data=0),
        dcc.Store(id="home-refresh-trigger", storage_type="memory", data=0),
        dcc.Store(id="config-refresh-trigger", storage_type="memory", data=0),
        dbc.Row(
            [
                dbc.Col(
                    html.H1("桥梁结构健康监测系统", className="text-center my-2"),
                    width=8
                ),
                dbc.Col(
                    [
                        html.Label("当前桥梁:", className="me-2"),
                        dcc.Dropdown(
                            id="global-bridge-selector",
                            placeholder="选择桥梁...",
                            style={"minWidth": "200px"}
                        ),
                    ],
                    width=4,
                    className="d-flex align-items-center justify-content-end"
                ),
            ],
            className="align-items-center"
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(sidebar, width=2),
                dbc.Col(dash.page_container, width=10),
            ]
        ),
        html.Div(id="global-notifications", className="mt-3"),
    ],
    fluid=True,
)


@callback(
    Output("global-bridge-selector", "options"),
    Input("url", "pathname"),
    Input("bridge-selector-trigger", "data"),
)
def update_global_bridge_selector(pathname, trigger):
    return get_bridge_options()


@callback(
    Output("current-bridge-store", "data"),
    Output("global-notifications", "children"),
    Output("bridge-selector-trigger", "data"),
    Input("global-bridge-selector", "value"),
    State("current-bridge-store", "data"),
    State("bridge-selector-trigger", "data"),
    prevent_initial_call=True,
)
def on_global_bridge_change(bridge_id, current_store, trigger):
    if bridge_id is None:
        return current_store, None, trigger
    
    bridge = Bridge.load(bridge_id)
    if bridge is None:
        return current_store, dbc.Alert("桥梁不存在", color="danger", duration=3000), trigger
    
    store_data = {
        "id": bridge.id,
        "name": bridge.name,
        "baseline_event_id": bridge.baseline_event_id
    }
    
    msg = dbc.Alert(
        f"已切换到桥梁: {bridge.name}",
        color="info",
        duration=2000
    )
    
    return store_data, msg, (trigger or 0) + 1


@callback(
    Output("global-bridge-selector", "value"),
    Input("current-bridge-store", "modifications"),
    State("current-bridge-store", "data"),
    State("global-bridge-selector", "value"),
    prevent_initial_call=True,
)
def sync_bridge_selector(modifications, store_data, current_value):
    if store_data and store_data.get("id"):
        if store_data["id"] != current_value:
            return store_data["id"]
    return dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)
