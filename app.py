import dash
from dash import Dash, html, dcc, Input, Output, State, callback, ALL, ctx
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
        dbc.NavLink("告警管理", href="/alert-management", active="exact"),
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
        dcc.Store(id="current-bridge-store", storage_type="local", data=None),
        dcc.Store(id="current-event-store", storage_type="local"),
        dcc.Store(id="baseline-store", storage_type="local"),
        dcc.Store(id="bridge-list-refresh", storage_type="memory", data=0),
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
    Output("global-bridge-selector", "value"),
    Input("url", "pathname"),
    Input("bridge-list-refresh", "data"),
    State("current-bridge-store", "data"),
)
def update_global_selector(pathname, refresh, store_data):
    options = get_bridge_options()
    bridge_id = store_data.get("id") if store_data else None
    if bridge_id:
        valid_ids = [o["value"] for o in options]
        if bridge_id not in valid_ids:
            bridge_id = None
    return options, bridge_id


@callback(
    Output("current-bridge-store", "data"),
    Output("global-notifications", "children"),
    Output("bridge-list-refresh", "data", allow_duplicate=True),
    Input("global-bridge-selector", "value"),
    State("current-bridge-store", "data"),
    State("bridge-list-refresh", "data"),
    prevent_initial_call=True,
)
def on_global_bridge_change(bridge_id, current_store, refresh):
    if bridge_id is None:
        return current_store, None, refresh

    bridge = Bridge.load(bridge_id)
    if bridge is None:
        return current_store, dbc.Alert("桥梁不存在", color="danger", duration=3000), refresh

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

    return store_data, msg, (refresh or 0) + 1


@callback(
    Output("current-bridge-store", "data", allow_duplicate=True),
    Output("bridge-list-refresh", "data", allow_duplicate=True),
    Input("home-bridge-selector", "value"),
    Input("config-bridge-selector", "value"),
    Input("import-bridge-selector", "value"),
    Input("modal-bridge-selector", "value"),
    Input("damage-bridge-selector", "value"),
    Input("monitoring-bridge-selector", "value"),
    Input("alert-mgmt-bridge-selector", "value"),
    Input("report-bridge-selector", "value"),
    State("current-bridge-store", "data"),
    State("bridge-list-refresh", "data"),
    prevent_initial_call=True,
)
def on_any_page_bridge_change(
    home_val, config_val, import_val, modal_val,
    damage_val, monitoring_val, alert_mgmt_val, report_val,
    current_store, refresh
):
    triggered_id = ctx.triggered_id
    if not triggered_id:
        return dash.no_update, dash.no_update

    mapping = {
        "home-bridge-selector": home_val,
        "config-bridge-selector": config_val,
        "import-bridge-selector": import_val,
        "modal-bridge-selector": modal_val,
        "damage-bridge-selector": damage_val,
        "monitoring-bridge-selector": monitoring_val,
        "alert-mgmt-bridge-selector": alert_mgmt_val,
        "report-bridge-selector": report_val,
    }

    bridge_id = mapping.get(triggered_id)
    if not bridge_id:
        return dash.no_update, dash.no_update

    bridge = Bridge.load(bridge_id)
    if bridge is None:
        return dash.no_update, dash.no_update

    store_data = {
        "id": bridge.id,
        "name": bridge.name,
        "baseline_event_id": bridge.baseline_event_id
    }

    return store_data, (refresh or 0) + 1


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
