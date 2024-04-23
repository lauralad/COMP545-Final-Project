from datetime import datetime
import json
import os
import time
from pathlib import Path

import streamlit as st

from utils import (
    load_json,
    load_json_no_cache,
    parse_arguments,
    format_chat_message,
    find_screenshot,
    gather_chat_history,
    get_screenshot,
    load_page,
)


def show_selectbox(demonstration_dir):
    # find all the subdirectories in the current directory
    dirs = [
        d
        for d in os.listdir(demonstration_dir)
        if os.path.isdir(f"{demonstration_dir}/{d}")
    ]

    if not dirs:
        st.title("No recordings found.")
        return None

    # sort by date
    dirs.sort(key=lambda x: os.path.getmtime(f"{demonstration_dir}/{x}"), reverse=True)

    # offer the user a dropdown to select which recording to visualize, set a default
    recording_name = st.sidebar.selectbox("Recording", dirs, index=0)

    return recording_name


def show_overview(data, recording_name, basedir):
    st.title('[WebLINX](https://mcgill-nlp.github.io/weblinx) Explorer')
    st.header(f"Recording: `{recording_name}`")

    screenshot_size = st.session_state.get("screenshot_size_view_mode", "regular")
    show_advanced_info = st.session_state.get("show_advanced_information", False)

    if screenshot_size == "regular":
        col_layout = [1.5, 1.5, 7, 3.5]
    elif screenshot_size == "small":
        col_layout = [1.5, 1.5, 7, 2]
    else:  # screenshot_size == 'large'
        col_layout = [1.5, 1.5, 11]

    # col_i, col_time, col_act, col_actvis = st.columns(col_layout)
    # screenshots = load_screenshots(data, basedir)

    for i, d in enumerate(data):
        # print("index", i, "data", d)
        st.write(f"index {i}, data {d}")

        if i > 0 and show_advanced_info:
            # Use html to add a horizontal line with minimal gap
            st.markdown(
                "<hr style='margin-top: 0.1rem; margin-bottom: 0.1rem;'/>",
                unsafe_allow_html=True,
            )
        if screenshot_size == "large":
            col_time, col_i, col_act = st.columns(col_layout)
            col_actvis = col_act
        else:
            col_time, col_i, col_act, col_actvis = st.columns(col_layout)
        secs_from_start = d["timestamp"] - data[0]["timestamp"]
        # `secs_from_start` is a float including ms, display in MM:SS.mm format
        col_time.markdown(
            f"**{datetime.utcfromtimestamp(secs_from_start).strftime('%M:%S')}**"
        )
        
        if not st.session_state.get("enable_html_download", True):
            col_i.markdown(f"**#{i}**")
        
        elif d["type"] == "browser" and (page_filename := d["state"]["page"]):
            page_path = f"{basedir}/pages/{page_filename}"

            col_i.download_button(
                label="#" + str(i),
                data=load_page(page_path),
                file_name=recording_name + "-" + page_filename,
                mime="multipart/related",
                key=f"page{i}",
            )
        else:
            col_i.button(f"#{i}", type='secondary')

        if d["type"] == "chat":
            col_act.markdown(format_chat_message(d), unsafe_allow_html=True)
            continue

        # screenshot_filename = d["state"]["screenshot"]
        img = get_screenshot(d, basedir)
        arguments = parse_arguments(d["action"])
        st.write(f"parsed arguments {arguments}")

        event_type = d["action"]["intent"]

        action_str = f"**{event_type}**({arguments})"

        if img:
            col_actvis.image(img)
        
        col_act.markdown(action_str)

        if show_advanced_info:
            status = d["state"].get("screenshot_status", "unknown")

            text = ""
            if status == "good":
                text += f'**:green[Used in demo]**\n\n'
            text += f'Screenshot: `{d["state"]["screenshot"]}`\\\n'
            text += f'Page: `{d["state"]["page"]}`\n'

            col_act.markdown(text)


def load_recording(basedir):
    # Before loading replay, we need a dropdown that allows us to select replay.json or replay_orig.json
    # Find all files in basedir starting with "replay" and ending with ".json"
    replay_files = sorted(
        [
            f
            for f in os.listdir(basedir)
            if f.startswith("replay") and f.endswith(".json")
        ]
    )
    replay_file = st.sidebar.selectbox("Select replay", replay_files, index=0)
    st.sidebar.checkbox(
        "Advanced Screenshot Info", False, key="show_advanced_information"
    )
    st.sidebar.checkbox(
        "Enable HTML download", False, key="enable_html_download"
    )
    replay_file = replay_file.replace(".json", "")
    
    if not Path(basedir).joinpath('metadata.json').exists():
        st.error(f"Metadata file not found at {basedir}/metadata.json. This is likely an issue with Huggingface Spaces. Try cloning this repo and running locally.")
        st.stop()
    
    metadata = load_json_no_cache(basedir, "metadata")

    # convert timestamp to readable date string
    recording_start_timestamp = metadata["recordingStart"]
    recording_start_date = datetime.fromtimestamp(
        int(recording_start_timestamp) / 1000
    ).strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.markdown(f"**started**: {recording_start_date}")

    # recording_end_timestamp = k["recordingEnd"]
    # calculate duration
    # duration = int(recording_end_timestamp) - int(recording_start_timestamp)
    # duration = time.strftime("%M:%S", time.gmtime(duration / 1000))

    # Read in the JSON data
    replay_dict = load_json_no_cache(basedir, replay_file)
    form = load_json_no_cache(basedir, "form")
    
    if replay_dict is None:
        st.error(f"Replay file not found at {basedir}/{replay_file}. This is likely an issue with Huggingface Spaces. Try cloning this repo and running locally.")
        st.stop()
    
    if form is None:
        st.error(f"Form file not found at {basedir}/form.json. This is likely an issue with Huggingface Spaces. Try cloning this repo and running locally.")
        st.stop()

    duration = replay_dict["data"][-1]["timestamp"] - replay_dict["data"][0]["timestamp"]
    duration = time.strftime("%M:%S", time.gmtime(duration))
    st.sidebar.markdown(f"**duration**: {duration}")

    if not replay_dict:
        return None

    for key in [
        "annotator",
        "description",
        "tasks",
        "upload_date",
        "instructor_sees_screen",
        "uses_ai_generated_output",
    ]:
        if form and key in form:
            # Normalize the key to be more human-readable
            key_name = key.replace("_", " ").title()

            if type(form[key]) == list:
                st.sidebar.markdown(f"**{key_name}**: {', '.join(form[key])}")
            else:
                st.sidebar.markdown(f"**{key_name}**: {form[key]}")

    st.sidebar.markdown("---")
    if replay_dict and "status" in replay_dict:
        st.sidebar.markdown(f"**Validation status**: {replay_dict['status']}")
    
    processed_meta_path = Path(basedir).joinpath('processed_metadata.json')
    start_frame = 'file not found'
    
    if processed_meta_path.exists():
        with open(processed_meta_path) as f:
            processed_meta = json.load(f)
        start_frame = processed_meta.get('start_frame', 'info not in file')
    
    st.sidebar.markdown(f"**Recording start frame**: {start_frame}")
    
    
    # st.sidebar.button("Delete recording", type="primary", on_click=delete_recording, args=[basedir])

    data = replay_dict["data"]
    return data


def run():
    # mode = st.sidebar.radio("Mode", ["Overview"])
    demonstration_dir = "./wl_data/demonstrations"

    # # params = st.experimental_get_query_params()
    # params = st.query_params
    # print(params)
    
    # # list demonstrations/
    # demo_names = os.listdir(demonstration_dir)

    # if params.get("recording"):
    #     if isinstance(params["recording"], list):
    #         recording_name = params["recording"][0]
    #     else:
    #         recording_name = params["recording"]

    # else:
    #     recording_name = demo_names[0]
    
    # recording_name = st.sidebar.selectbox(
    #     "Recordings",
    #     demo_names,
    #     index=demo_names.index(recording_name),
    # )

    # if recording_name != params.get("recording", [None])[0]:
    #     # st.experimental_set_query_params(recording=recording_name)
    #     # use st.query_params as a dict instead
    #     st.query_params['recording'] = recording_name


    demo_names = os.listdir(demonstration_dir)

    def update_recording_name():
        st.query_params["recording"] = st.session_state.get("recording_name", demo_names[0])

    # For initial run, set the query parameter to the selected recording
    if not st.query_params.get("recording"):
        update_recording_name()
    
    recording_name = st.query_params.get("recording")
    if recording_name not in demo_names:
        st.error(f"Recording `{recording_name}` not found. Please select another recording.")
        st.stop()
    
    recording_idx = demo_names.index(recording_name)
    st.sidebar.selectbox(
        "Recordings", demo_names, on_change=update_recording_name, key="recording_name", index=recording_idx
    )

    with st.sidebar:
        # Want a dropdown
        st.selectbox(
            "Screenshot size",
            ["small", "regular", "large"],
            index=1,
            key="screenshot_size_view_mode",
        )

    if recording_name is not None:
        basedir = f"{demonstration_dir}/{recording_name}"
        data = load_recording(basedir=basedir)

        if not data:
            st.stop()

        show_overview(data, recording_name=recording_name, basedir=basedir)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    run()