"""Read-only Streamlit browser for QWIP datastores."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qwip_explorer.config import ExplorerDefaults
from qwip_explorer.connections import close_datastore, connect_datastore
from qwip_explorer.core import flatten_frame, heatmap_figure, line_figure, plottable_columns
from qwip_explorer.state import clear_connection_results
from qwip_explorer.views import render_commit_history, render_dataset_browser, render_direct_lookup


st.set_page_config(page_title="QWIP Datastore Explorer", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; max-width: 1500px;}
    [data-testid="stMetricValue"] {font-size: 1.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


defaults = ExplorerDefaults.from_environment()


def show_plot_controls(
    frame_name: str,
    df: pd.DataFrame,
    *,
    key_prefix: str = "plot",
) -> None:
    flat, axes = flatten_frame(df)
    values = plottable_columns(flat, axes)
    widget_prefix = f"{key_prefix}_{frame_name}"

    st.caption(
        f"{len(df):,} rows · {len(df.columns):,} dependent columns · "
        f"axes: {', '.join(axes)}"
    )
    if not values:
        st.info("This table has no numeric dependent columns to plot.")
        st.dataframe(flat, use_container_width=True, height=430)
        return

    control_a, control_b, control_c, control_d = st.columns(4)
    default_dimension = "2D heatmap" if len(axes) >= 2 else "1D line"
    plot_kind = control_a.selectbox(
        "Plot type",
        ["1D line", "2D heatmap"],
        index=1 if default_dimension.startswith("2D") else 0,
        key=f"kind_{widget_prefix}",
    )
    x = control_b.selectbox(
        "X axis", axes, index=min(len(axes) - 1, 1), key=f"x_{widget_prefix}"
    )
    value = control_c.selectbox(
        "Dependent value", values, key=f"value_{widget_prefix}"
    )
    is_complex = pd.api.types.is_complex_dtype(flat[value].dtype)
    transforms = ["Magnitude", "Phase", "Real", "Imaginary"] if is_complex else ["Raw"]
    transform = control_d.selectbox(
        "Value display", transforms, key=f"transform_{widget_prefix}"
    )

    try:
        if plot_kind == "2D heatmap":
            if len(axes) < 2:
                st.info("A heatmap needs two independent index axes.")
                return
            y_options = [axis for axis in axes if axis != x]
            y = st.selectbox("Y axis", y_options, key=f"y_{widget_prefix}")
            figure = heatmap_figure(flat, x=x, y=y, value=value, transform=transform)
        else:
            color_options = ["None"] + [axis for axis in axes if axis != x]
            color_choice = st.selectbox(
                "Group/color by",
                color_options,
                key=f"color_{widget_prefix}",
                help="Use another swept parameter to draw one trace per value.",
            )
            figure = line_figure(
                flat,
                x=x,
                value=value,
                transform=transform,
                color=None if color_choice == "None" else color_choice,
            )
        st.plotly_chart(figure, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not build this plot: {exc}")

    with st.expander("Show plotted data"):
        st.dataframe(flat, use_container_width=True, height=430)
        st.download_button(
            "Download CSV",
            flat.to_csv(index=False).encode("utf-8"),
            file_name=f"{frame_name.replace(' / ', '_')}.csv",
            mime="text/csv",
            key=f"csv_{widget_prefix}",
        )


st.title("QWIP Datastore Explorer")
st.caption("Read-only dataset search, provenance inspection, and automatic 1D/2D plotting")

with st.sidebar:
    st.header("Connection")
    storage_options = ["HTTP", "Local folder"]
    preferred_storage_mode = defaults.storage_mode
    storage_mode = st.radio(
        "Asset storage",
        storage_options,
        index=(
            storage_options.index(preferred_storage_mode)
            if preferred_storage_mode in storage_options
            else 0
        ),
        horizontal=True,
        key="storage_mode",
    )
    with st.form("connection_form"):
        host = st.text_input("Database host", defaults.host)
        datastore_database = st.text_input(
            "Datastore database", defaults.datastore_database
        )
        configuration_database = st.text_input(
            "Configuration database",
            defaults.configuration_database,
            help="This is the ConfigDB used by notebooks and calibration commits.",
        )
        username = st.text_input(
            "Username",
            value=defaults.username,
            key="connection_username",
        )
        password = st.text_input("Password", defaults.password, type="password")
        default_storage = defaults.http_storage_url or (f"http://{host}:4002" if host else "")
        if storage_mode == "Local folder":
            default_storage = defaults.local_storage_directory
        storage_location = st.text_input(
            "Storage location", default_storage, key=f"storage_location_{storage_mode}"
        )
        connect_clicked = st.form_submit_button(
            "Connect datastore", type="primary", use_container_width=True
        )

if connect_clicked:
    st.session_state.credentials_submitted = True
    close_datastore(st.session_state.pop("datastore", None))
    clear_connection_results(st.session_state)
    try:
        with st.spinner("Connecting to QWIP datastore..."):
            st.session_state.datastore = connect_datastore(
                host,
                datastore_database,
                username,
                password,
                storage_mode,
                storage_location,
            )
            st.session_state.connected_username = username
            st.session_state.connected_datastore_database = datastore_database
            st.session_state.connected_configuration_database = configuration_database
        st.success("Connected.")
    except Exception as exc:
        close_datastore(st.session_state.pop("datastore", None))
        st.error(f"Connection failed: {exc}")

datastore = st.session_state.get("datastore")
active_database = st.session_state.get("connected_datastore_database", datastore_database)
active_configuration_database = st.session_state.get(
    "connected_configuration_database", configuration_database
)
active_username = st.session_state.get("connected_username", username)

main_view = st.segmented_control(
    "Explorer view",
    ["Datastore Explorer", "Configuration history"],
    default="Datastore Explorer",
    label_visibility="collapsed",
    key="main_view",
)

if main_view == "Datastore Explorer":
    st.caption(f"Dataset catalog: `{active_database}`")
    if datastore is None:
        st.info(
            "Enter the credentials in the sidebar and select **Connect datastore** "
            "to search datasets or open a dataset by ID."
        )
    else:
        dataset_view = st.segmented_control(
            "Datastore view",
            ["Search datasets", "Open by dataset ID"],
            default="Search datasets",
            label_visibility="collapsed",
            key="dataset_view",
        )
        if dataset_view == "Search datasets":
            render_dataset_browser(datastore, show_plot_controls)
        else:
            render_direct_lookup(datastore, show_plot_controls)

else:
    st.caption(f"Notebook ConfigDB: `{active_configuration_database}`")
    if not username.strip() or not password:
        st.info(
            "Enter the database credentials in the sidebar. Configuration history "
            "uses them independently from the datastore connection."
        )
    else:
        render_commit_history(
            host=host,
            datastore_database=active_database,
            configuration_database=active_configuration_database,
            username=active_username,
            password=password,
        )
