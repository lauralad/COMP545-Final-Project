from datetime import datetime
import json
import os
import time
from pathlib import Path
from PIL import Image
# from playwright.sync_api import sync_playwright
import subprocess
import streamlit as st
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
import os


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

from playwright.sync_api import sync_playwright, Playwright, Browser
from datasets import load_dataset
import base64

playwright: Playwright = None
browser: Browser = None
page = None
# dataset_splits = ["test_cat", "test_geo", "test", "test_vis", "test_web", "train", "validation"]
dataset_splits = ["validation"]
datasets = {}
unique_data_dict = {}


@st.cache(allow_output_mutation=True)
def load_and_prepare_data():
    global datasets
    # datasets = {}
    unique_data_dict = {}
    for split in dataset_splits:
        datasets[split] = load_dataset("McGill-NLP/weblinx", split=split)

    for split, dataset in datasets.items():
        unique_data_dict[split] = {}
        for turn in dataset:
            demo = turn["demo"]
            turn_num = turn["turn"]
            if demo not in unique_data_dict[split]:
                unique_data_dict[split][demo] = {}
            if turn_num not in unique_data_dict[split][demo]:
                unique_data_dict[split][demo][turn_num] = turn
    return unique_data_dict

def setup_datasets():
    global unique_data_dict
    unique_data_dict = load_and_prepare_data()

def setup_browser():
    global playwright, browser, page
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()


def load_html_content(html_content):
    """Loads the given HTML content in the Playwright page."""
    # Convert HTML content to a data URL
    encoded_html = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
    data_url = f"data:text/html;base64,{encoded_html}"
    page.goto(data_url)

def shutdown_browser():
    if browser:
        browser.close()
    if playwright:
        playwright.stop()

def install_playwright():
    # Install Playwright and browsers
    # subprocess.run(["pip", "install", "playwright"], check=True)
    os.system('playwright install-deps')
    # Install browsers used by Playwright
    os.system('playwright install')

def execute_browser_action(action):
    
    screenshot_path = "screenshot.png"
    
    if action['intent'] == 'load':
        page.goto(action['arguments']['metadata']['url'])
    elif action['intent'] == 'click':
        page.click(action['arguments']['element']['xpath'])  # assuming xpath is always available
    elif action['intent'] == 'textInput':
        page.fill(action['arguments']['element']['xpath'], action['text'])
    elif action['intent'] == 'paste':
        # Simulate paste action (Playwright doesn't have a direct paste method)
        page.type(action['arguments']['element']['xpath'], action['pasted'])
    elif action['intent'] == 'submit':
        page.query_selector(action['arguments']['element']['xpath']).evaluate("element => element.submit()")
    page.screenshot(path=screenshot_path)
    
    return screenshot_path

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


def show_overview(data, recording_name, dataset, demo_name, turn, basedir):
    st.title('[WebLINX](https://mcgill-nlp.github.io/weblinx) Explorer')
    st.header(f"Recording: `{dataset} > Demo {demo_name} > Turn {turn}`")
    screenshot_path = "screenshot.png"
    # Find indices for instructor chat turns
    instructor_turns = [i for i, d in enumerate(data) if d['type'] == 'chat' and d['speaker'] == 'instructor']
    # st.write(f"instructor_turns {instructor_turns}")
    # st.write(f"turn 5 {data[5]}")

    selected_turn_idx = turn #6
    screenshot_size = st.session_state.get("screenshot_size_view_mode", "regular")
    show_advanced_info = st.session_state.get("show_advanced_information", False)

    # if screenshot_size == "regular":
    #     col_layout = [1.5, 1.5, 7, 3.5]
    # elif screenshot_size == "small":
    #     col_layout = [1.5, 1.5, 7, 2]
    # else:  # screenshot_size == 'large'
    #     col_layout = [1.5, 1.5, 11]

    if screenshot_size == "regular":
        col_layout = [1.5, 1.5, 3.5, 3.5]  # Adjusted for two columns
    elif screenshot_size == "small":
        col_layout = [1.5, 1.5, 3.5, 3.5]
    else:  # screenshot_size == 'large'
        col_layout = [1.5, 1.5, 5.5, 5.5]


    # col_i, col_time, col_act, col_actvis = st.columns(col_layout)
    # screenshots = load_screenshots(data, basedir)

    # Find the last instructor turn index before the selected turn index
    previous_instructor_turn_idx = max([idx for idx in instructor_turns if idx < turn], default=None)
    

    if previous_instructor_turn_idx is None:
        st.write("No previous instructor turn found.")
        return  # Exit the function if there's no previous instructor turn

    for i in range(previous_instructor_turn_idx, turn + 1):
        d = data[i]
        
        if i > 0 and show_advanced_info:
            # Use html to add a horizontal line with minimal gap
            st.markdown(
                "<hr style='margin-top: 0.1rem; margin-bottom: 0.1rem;'/>",
                unsafe_allow_html=True,
            )
      
        # Create columns for the overview
        cols = st.columns(col_layout)
        col_time, col_i, col_act1, col_act2 = cols  # Split into two action columns
        # col_time, col_i, col_act, col_actvis = st.columns(col_layout)
        # Adding titles for each column
        col_act1.markdown("### True Answer")
        col_act2.markdown("### Model Prediction")
        

        secs_from_start = d["timestamp"] - data[0]["timestamp"] #data
        # `secs_from_start` is a float including ms, display in MM:SS.mm format
        col_time.markdown(
            f"**{datetime.utcfromtimestamp(secs_from_start).strftime('%M:%S')}**"
        )
        
        if not st.session_state.get("enable_html_download", True):
            col_i.markdown(f"**#{i}**")
        
        elif d["type"] == "browser" and (page_filename := d["state"]["page"]):
            page_path = f"{basedir}/pages/{page_filename}"
            html_content = load_page(page_path)
            load_html_content(html_content)

            # html_page = load_page(page_path)
            # load_html_content(html_page)
            page.screenshot(path="screenshot.png")

            col_i.download_button(
                label="#" + str(i),
                data=load_page(page_path), #data
                file_name=recording_name + "-" + page_filename,
                mime="multipart/related",
                key=f"page{i}",
            )
        else:
            col_i.button(f"#{i}", type='secondary')

        if d["type"] == "chat":
            # col_act.markdown(format_chat_message(d), unsafe_allow_html=True)
            col_act1.markdown(format_chat_message(d), unsafe_allow_html=True)
            col_act2.markdown(format_chat_message(d), unsafe_allow_html=True)
            continue

        # screenshot_filename = d["state"]["screenshot"]
        img = get_screenshot(d, basedir)
        arguments = parse_arguments(d["action"])
        # st.write(f"parsed arguments {arguments}")

        event_type = d["action"]["intent"]

        action_str = f"**{event_type}**({arguments})"

        # Do the action in the browser
        # execute_browser_action(d['action'])
        # screenshot_path = execute_browser_action(d['action'])
        if screenshot_path:
            img = Image.open(screenshot_path)
            col_act2.image(img, caption="Screenshot after action")
        if img:
            col_act1.image(img)
        
        # col_act.markdown(action_str)
        col_act1.markdown(action_str)
        col_act2.markdown(action_str)

        if show_advanced_info:
            status = d["state"].get("screenshot_status", "unknown")

            text = ""
            if status == "good":
                text += f'**:green[Used in demo]**\n\n'
            text += f'Screenshot: `{d["state"]["screenshot"]}`\\\n'
            text += f'Page: `{d["state"]["page"]}`\n'

            col_act1.markdown(text)
            col_act2.markdown(text)


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
    # st.sidebar.checkbox(
    #     "Enable HTML download", False, key="enable_html_download"
    # )
    replay_file = replay_file.replace(".json", "")
    
    if not Path(basedir).joinpath('metadata.json').exists():
        st.error(f"Metadata file not found at {basedir}/metadata.json. This is likely an issue with Huggingface Spaces. Try cloning this repo and running locally.")
        st.stop()
    
    metadata = load_json_no_cache(basedir, "metadata")

    # convert timestamp to readable date string
    # recording_start_timestamp = metadata["recordingStart"]
    # recording_start_date = datetime.fromtimestamp(
    #     int(recording_start_timestamp) / 1000
    # ).strftime("%Y-%m-%d %H:%M:%S")
    # st.sidebar.markdown(f"**started**: {recording_start_date}")

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

    # duration = replay_dict["data"][-1]["timestamp"] - replay_dict["data"][0]["timestamp"]
    # duration = time.strftime("%M:%S", time.gmtime(duration))
    # st.sidebar.markdown(f"**duration**: {duration}")

    # if not replay_dict:
    #     return None

    # for key in [
    #     "annotator",
    #     "description",
    #     "tasks",
    #     "upload_date",
    #     "instructor_sees_screen",
    #     "uses_ai_generated_output",
    # ]:
    #     if form and key in form:
    #         # Normalize the key to be more human-readable
    #         key_name = key.replace("_", " ").title()

    #         if type(form[key]) == list:
    #             st.sidebar.markdown(f"**{key_name}**: {', '.join(form[key])}")
    #         else:
    #             st.sidebar.markdown(f"**{key_name}**: {form[key]}")

    st.sidebar.markdown("---")
    # if replay_dict and "status" in replay_dict:
    #     st.sidebar.markdown(f"**Validation status**: {replay_dict['status']}")
    
    processed_meta_path = Path(basedir).joinpath('processed_metadata.json')
    start_frame = 'file not found'
    
    if processed_meta_path.exists():
        with open(processed_meta_path) as f:
            processed_meta = json.load(f)
        start_frame = processed_meta.get('start_frame', 'info not in file')
    
    # st.sidebar.markdown(f"**Recording start frame**: {start_frame}")
    
    
    # st.sidebar.button("Delete recording", type="primary", on_click=delete_recording, args=[basedir])

    data = replay_dict["data"]
    return data


def run():
    
    # print("splits", datasets)
    setup_browser()
    try:
    
        demonstration_dir = "./wl_data/demonstrations"
        
        demo_names = os.listdir(demonstration_dir)

        # def update_recording_name():
        #     st.query_params["recording"] = st.session_state.get("recording_name", demo_names[0])

        # # For initial run, set the query parameter to the selected recording
        # if not st.query_params.get("recording"):
        #     update_recording_name()
        
        # recording_name = st.query_params.get("recording")
        # if recording_name not in demo_names:
        #     st.error(f"Recording `{recording_name}` not found. Please select another recording.")
        #     st.stop()
        
        # recording_idx = demo_names.index(recording_name)
        # st.sidebar.selectbox(
        #     "Recordings", demo_names, on_change=update_recording_name, key="recording_name", index=recording_idx
        # )
        dataset = st.sidebar.selectbox("Select Dataset", list(unique_data_dict.keys()))
        if dataset:
            demos = unique_data_dict[dataset]
            demo_name = st.sidebar.selectbox("Select Demo", list(demos.keys()))

            # Dropdown to select the turn number based on the selected demo
            if demo_name:
                turns = demos[demo_name]
                selected_turn = st.sidebar.selectbox("Select Turn Number", sorted(turns))

                # Display or process data based on selected turn
                # st.write(f"Selected Dataset: {dataset}")
                # st.write(f"Selected Demo: {demo_name}")
                # st.write(f"Selected Turn: {selected_turn}")

                

                with st.sidebar:
                    # Want a dropdown
                    st.selectbox(
                        "Screenshot size",
                        ["small", "regular", "large"],
                        index=1,
                        key="screenshot_size_view_mode",
                    )

                if dataset is not None and demo_name is not None and selected_turn is not None:
                    recording_name =  demo_name
                    basedir = f"{demonstration_dir}/{recording_name}"
                    data = load_recording(basedir=basedir)

                    # data = unique_data_dict[dataset][demo_name][selected_turn]

                    show_overview(data, recording_name=recording_name,dataset=dataset, demo_name=demo_name, turn=selected_turn, basedir=basedir)
    finally:
        shutdown_browser()
    


if __name__ == "__main__":
    install_playwright()
    setup_datasets()
    # st.write("unique tuples", unique_data_dict)
    # st.set_page_config(layout="wide")
    run()