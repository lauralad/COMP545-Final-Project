from datetime import datetime
import json
import os
import time
from pathlib import Path
from PIL import Image
import subprocess
import streamlit as st
import os
import pickle

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

from playwright.sync_api import sync_playwright, Playwright, Browser, TimeoutError
from datasets import load_dataset
import base64
import pandas as pd
import re
import torch
from transformers import pipeline
from huggingface_hub import snapshot_download
from weblinx.processing import load_candidate_elements
from bs4 import BeautifulSoup

# Global variables
action_model = None
template = None

playwright: Playwright = None
browser: Browser = None
page = None
# dataset_splits = ["test_cat", "test_geo", "test", "test_vis", "test_web", "train", "validation"]
dataset_splits = ["validation"]
datasets = {}
unique_data_dict = {}
cleaned_data = [] #json of preds
data_mapping = {}
# valid_candidates_list = load_candidate_elements("./wl_data/candidates/valid.jsonl")
pred_mapping = {}

def init_model():
    global action_model, template

    # Ensure the right device is used (CPU or CUDA)
    device = 0 if torch.cuda.is_available() else -1

    # Download templates
    snapshot_download(
        "McGill-NLP/WebLINX", repo_type="dataset", allow_patterns="templates/*", local_dir="."
    )

    # Load the template
    with open('templates/llama.txt') as f:
        template = f.read()

    # Load the model
    action_model = pipeline(
        model="McGill-NLP/Sheared-LLaMA-2.7B-weblinx", device=device, torch_dtype='auto'
    )

def get_pred_for_turn(split, demo, turn_num):
    turn = unique_data_dict[split][demo][turn_num]
    turn_formatted = template.format(**turn)
    # action_model is the llama model of choice
    out = action_model(turn_formatted, return_full_text=False, 
                       max_new_tokens=64, truncation=True)
    pred = out[0]['generated_text']
    closing_paren_index = pred.find(')')
    substring = pred[:closing_paren_index + 1]
    pred_cleaned = substring.strip()
    return pred_cleaned

#new function that can grab the uid out of a string
def extract_uid(action):
    action_pattern = re.compile(r'\b\w+\(.*?\buid="([\w-]+)"\)')
    uid_match = action_pattern.search(action)
    if uid_match:
        return uid_match.group(1)
    else:
        return None
    
# def translate_pred(uid, demo, turn):
#     turn_cands = valid_candidates_list[(demo, int(turn))]
#     pred_uid = uid
#     pred_xpath = None
#     pred_class = None
#     for i in turn_cands:
#         if i['uid'] == pred_uid:
#             pred_xpath = i["elem_dict"]["xpath"]
#             pattern = r"class='([^']+)'"
#             elem_attributes = i["elem_dict"]["attributes"]
#             pred_class = re.search(pattern, elem_attributes).group(1)
#     return pred_class, pred_xpath



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

# def clean_json_file(file_path):
#     # Read the JSON data from file
#     with open(file_path, 'r') as file:
#         pred_data = json.load(file)

#     # Clean up each string in the list
#     cleaned_data = []
#     for item in pred_data:
#         # Strip leading/trailing whitespace and reduce any internal excess whitespace
#         cleaned_item = ' '.join(item.split())
#         cleaned_data.append(cleaned_item)

#     return cleaned_data

def clean_json_file(file_path):
    # Read the JSON data from file
    with open(file_path, 'r') as file:
        pred_data = json.load(file)
    cleaned_data = []
    for i, item in enumerate(pred_data):
        #grab first action only
        open_paren = 0 
        close_paren = 0 
        last_balanced_index = 0
        initial_split = ' '.join(item.split())
        for index, char in enumerate(initial_split):
            if char == '(':
                open_paren += 1
            elif char == ')':
                close_paren += 1
            if open_paren == close_paren:
                last_balanced_index = index + 1 
        if open_paren == close_paren:
            substring = initial_split[:last_balanced_index]
        else:
            substring = initial_split
        cleaned_item = ' '.join(substring.split())
        cleaned_data.append(cleaned_item)
    return cleaned_data

def load_csv_data(file_path):
    # Read the CSV using Pandas and extract demo names and turn numbers
    pred_df = pd.read_csv(file_path)
    return pred_df[['demo', 'turn']]

def interpret_action(action):
    action_type = action.split("(")[0].strip()
    output = None
    if action_type == "click":
        output = extract_uid(action)
    elif action_type == "text_input":
        action_uid = extract_uid(action)
        input_pattern = r'text="([^"]*)'
        input_match = re.search(input_pattern, action)
        text_input = None
        if input_match:
            text_input = input_match.group(1)
        else: 
            print("input_match:", input_match, "action:", action)
            print(text_input)
            raise KeyError
        output = [action_uid, text_input]
        
    elif action_type == "say":
        utterance_pattern = r'utterance="([^"]*)'
        match = re.search(utterance_pattern, action)
        if match:
            output = match.group(1)
        else:
            print("match:", match)
            print(print(action))
            raise KeyError
        
    elif action_type == "submit":
        output = extract_uid(action)
        
    elif action_type == "load":
        url_pattern = r'url="([^"]*)'
        url_match = re.search(url_pattern, action)
        if url_match:
            output = url_match.group(1)
        else:
            print(action)
            raise KeyError
        
        
    elif action_type == "scroll":
        scroll_pattern = re.compile(r'scroll\(x=(\d+), y=(\d+)\)')
        match = scroll_pattern.match(action)
        
        if match:
            x_value = int(match.group(1))
            y_value = int(match.group(2))
            output = [x_value, y_value]
    elif action_type == "change":
        action_uid = extract_uid(action)
        change_pattern = r'value="([^"]*)'
        change_match = re.search(change_pattern, action)
        if change_match:
            output = change_match.group(1)
        else:
            print(action)
            raise KeyError
    else:
        pass  
    return action_type, output   

def create_mapping(csv_df):
    global data_mapping
    i = 0
    for index, row in csv_df.iterrows():
        demo_name = row['demo']
        turn_number = row['turn']
        key = f"{demo_name}_{turn_number}"  # Create a string key
        data_mapping[key] = index
        
        
    return data_mapping

def get_preds():
    global pred_mapping
    with open('preds.pkl', 'rb') as f:
        bad_mapping = pickle.load(f)
    #Warning: this data structure was not fully serializable as JSON due to one or more unexpected keys. (Error was: keys must be str, int, float, bool or None, not tuple)
    pred_mapping = {str(key): value for key, value in bad_mapping.items()}
    return pred_mapping

def extract_attributes(html):
    # Parse the HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find the input element (assuming there's only one input in the HTML snippet)
    input_element = soup.find('input')
    
    # Extract id and class attributes
    element_id = input_element.get('id', None)
    element_classes = input_element.get('class', [])  # This returns a list of classes
    
    return element_id, element_classes

def extract_non_say_actions(df, demo_name, turn_number):
    # df is the unique_data_dict
    # Retrieve the row based on demo_name and turn_number
    row = df[demo_name][turn_number]
    
    action_history = row['action_history']
    # Regex to find non-"say" actions
    # actions = re.findall(r'(?<!say)\(.*?\)', action_history)
    actions = re.findall(r'\b(?!say\b)\w+\(.*?\)', action_history)
    return actions


def get_browser_actions_up_to_turn(data, demo_name, turn_number):
    # Initialize an empty list to store browser actions
    browser_actions = []

    # Loop through each entry in the data up to and including the specified turn
    for i, d in enumerate(data):
        # Stop collecting once you pass the desired turn number
        if i > turn_number:
            break
        # Check if the entry is of type 'browser'
        if d["type"] == "browser":
            # Parse the action arguments and get the event type
            arguments = parse_arguments(d["action"])
            event_type = d["action"]["intent"]
            
            # Append the action information as a tuple or dictionary
            browser_actions.append((event_type, arguments))
    return browser_actions
    
def parse_action_details(action):
    match = re.match(r'(\w+)\((\w+)="([^"]+)"\)', action)
    if match:
        return {
            'function': match.group(1),
            'argument': match.group(2),
            'value': match.group(3)
        }
    return {}

def setup_datasets():
    global unique_data_dict, cleaned_data, data_mapping, pred_mapping
    file_path = './valid_predictions.json'
    # clean_json_file has been updated to return not just the action, but also the action type of that action, if you need it.
    #format is [pred_action, pred_action's type]
    cleaned_data = clean_json_file(file_path)
    csv_df = load_csv_data("./valid.csv")
    data_mapping = create_mapping(csv_df)
    # pred_map is a dictionary, where the key is (demo_name, turn_num), and 
    # the value is a list: [predicted_action, [predicted_action's type, what the action says to do], predicted_action's uid (if available), [predicted candidate's class, predicted candidate's xpath] (if available)]
    # pred_map should be used to get the info that you need for the specific demo_name and turn_num that you need when selected.
    # Example below:
    # full action: [0], the action type is text_input [1][0], the uid of the object it's doing it to is 67e2a5fb-8b1d-41a0 ([1][1][0]) the text is biotechnlogy ([1][1][1]), the uid of the action is [2]
    # the class of the object is [3][0], the xpath is [3][1]. for different actions these will be slightly different, specifically how [1] looks. 
    # ['text_input(text="biotechnology", uid="67e2a5fb-8b1d-41a0")', ['text_input', ['67e2a5fb-8b1d-41a0', 'biotechnology']], '67e2a5fb-8b1d-41a0', ['searchbox form-search form-input', '/html/body/div[2]/div/div/div[1]/div[2]/div[3]/form/div[1]/input']]
    
    # if the predicted candidate's class is "get", it means that it's a submit action and it should do something along the lines of click enter. Example of what one of these looks like right below
    # ['submit(uid="c7fbc11c-0949-4ab2")', -['submit', 'c7fbc11c-0949-4ab2'], 'c7fbc11c-0949-4ab2', ['get', '/html/body/div[2]/div/div/div[1]/div[2]/div[3]/form']]
    pred_mapping = get_preds()
    st.write(pred_mapping)
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

def draw_click_marker(bounding_box):
    # Calculate the center of the element
    x = bounding_box['x'] + bounding_box['width'] / 2
    y = bounding_box['y'] + bounding_box['height'] / 2
    
    # Draw a visual marker using JavaScript
    page.evaluate('''
        ({x, y}) => {
            const circle = document.createElement('div');
            circle.style.position = 'absolute';
            circle.style.borderRadius = '50%';
            circle.style.background = 'red';
            circle.style.width = '20px';
            circle.style.height = '20px';
            circle.style.left = `${x - 10}px`;
            circle.style.top = `${y - 10}px`;
            circle.style.zIndex = '10000';
            document.body.appendChild(circle);

            // Optional: Remove the circle after some time
            setTimeout(() => document.body.removeChild(circle), 1000);
        }
    ''', {'x': x, 'y': y})

    st.write(f"Marked click position at ({x}, {y})")

def execute_action(predicted_action):
    action_type = predicted_action[1][0]
    action_uid = predicted_action[2]
    xpath = predicted_action[3][1]

    screenshot_path = "screenshot.png"
    intent = action_type
    # intent = action['action']['intent']
    # args = action['action']['arguments']
    if intent == 'load':
        url = action_class
        page.goto(url)
        # print(f"Loaded URL: {url}")
        st.write(f"Loaded URL: {url}")
    elif intent == 'click':
        
        action_class = predicted_action[3][0]
        # st.write(action_type)
        # st.write(action_class)
        # xpath = "/html/body/div[2]/div/div/div[1]/div[2]/div[3]/form/div[1]/input"
    
        try:
            # Use the XPath to click the element
            page.click(f'xpath={xpath}', timeout=5000)  # Timeout after 5000 ms
            # st.success(f"Successfully clicked on the element at XPath '{xpath}'.")
        except TimeoutError:
            pass
            # st.error(f"Failed to click on the element at XPath '{xpath}' within 5 seconds.")
        element = page.locator(f'xpath={xpath}')
        # bounding_box = element.bounding_box()
        try:
            # Attempt to get bounding box and draw click marker
            bounding_box = element.bounding_box()
            if bounding_box:
                draw_click_marker(bounding_box)
        except TimeoutError:
            pass
            # st.warning("Timeout occurred while trying to get bounding box. Continuing without drawing marker.")
        
    elif intent == 'text_input':
        text = predicted_action[1][1][1]
        # st.write(f"Text: {text}")
        try:
            page.fill(f'xpath={xpath}', text)
            # st.success(f"Filled in the text '{text}' at the element with XPath '{xpath}'.")
            page.screenshot(path=screenshot_path)
        except TimeoutError:
            pass
            # st.error(f"Failed to click on the element at XPath '{xpath}' within 5 seconds.")

    elif intent == 'submit':
        # page.query_selector(xpath).evaluate("element => element.submit()")
        form_locator = page.locator(f'xpath={xpath}')
        
        # Check if the form is found and visible
        if form_locator.count() > 0:
            # Submit the form by evaluating JavaScript on the form element
            form_locator.evaluate("form => form.submit()")
            
            # Optionally, you might want to wait for a navigation event if the form submission leads to a new page
            # page.wait_for_navigation(timeout=5000)  # Waits for 5 seconds for navigation to complete
            
            # st.success("Form submitted successfully.")
        else:
            pass
            # st.error("No form found with the specified XPath.")
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

def filter_browser_actions(data, turn_index):
    browser_actions = []
    for i, entry in enumerate(data):
        if i == turn_index:
            break
        if entry['type'] == 'browser':
            browser_actions.append(entry)
    return browser_actions

def execute_browser_actions(browser_actions):
    for action in browser_actions:
        intent = action['action']['intent']
        args = action['action']['arguments']
        if intent == 'load':
            url = args['properties']['url']
            page.goto(url)
            # print(f"Loaded URL: {url}")
            st.write(f"Loaded URL: {url}")
        elif intent == 'click':
            x = args['metadata']['mouseX']
            y = args['metadata']['mouseY']
            page.mouse.click(x, y)
            st.write(f"Clicked at ({x}, {y})")
            # print(f"Clicked at ({x}, {y})")
            page.wait_for_timeout(1000)  # Wait for 1 second for demonstration
        elif intent == "paste":
            # st.write(args)
            elem_html = args['element']['outerHTML']
            element_id, element_classes = extract_attributes(elem_html)
            # st.write(f"Element ID: {element_id}")
            # st.write(f"Element Classes: {element_classes}")
            # xpath_expression = f"xpath=//{class_name}"
            try:
                # First try clicking by ID
                if element_id:
                    page.click(f"#{element_id}:visible")
                    page.fill(f"#{element_id}:visible", args['pasted'])
                    # print(f"Clicked using ID: #{element_id}")
                    st.write(f"Clicked using ID: #{element_id}")
                else:
                    raise TimeoutError("No ID provided, trying class selectors.")

            except TimeoutError as e:
                # print(f"Failed to click using ID: {str(e)}")
                st.write(f"Failed to click using ID: {str(e)}")
                # If ID click fails, try clicking by class
                for class_name in element_classes:
                    try:
                        page.click(f".{class_name}:visible")
                        # print(f"Clicked using class: .{class_name}")
                        st.write(f"Clicked using class: .{class_name}")
                        break
                    except TimeoutError:
                        # print(f"Failed to click using class: .{class_name}")
                        st.write(f"Failed to click using class: .{class_name}")
                        continue
                else:
                    # print("All class selectors failed.")
                    st.write("All class selectors failed.")

            # Optional: Wait to observe the effects
            # page.wait_for_timeout(1000)
            # if element_id:
            #     page.click(f"#{element_id}")  # CSS ID selector
            # Using the locator with the XPath to click the element
            # page.locator(xpath_expression).click()
            # page.click(f".{class_name}")
            # page.fill()

    

def show_overview(data, model_name, recording_name, dataset, demo_name, turn, basedir):
    st.title('[WebLINX](https://mcgill-nlp.github.io/weblinx) Explorer')
    st.header(f"Recording: `{dataset} > Demo {demo_name} > Turn {turn}`")
    screenshot_path = "screenshot.png"
    # Find indices for instructor chat turns
    instructor_turns = [i for i, d in enumerate(data) if d['type'] == 'chat' and d['speaker'] == 'instructor']
    
    browser_actions = filter_browser_actions(data, turn)
    execute_browser_actions(browser_actions)


    # st.write(details_list)

    selected_turn_idx = turn #6
    screenshot_size = st.session_state.get("screenshot_size_view_mode", "regular")
    show_advanced_info = st.session_state.get("show_advanced_information", False)

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
    browser_actions = get_browser_actions_up_to_turn(data, demo_name, turn)

    if previous_instructor_turn_idx is None:
        st.write("No previous instructor turn found.")
        return  # Exit the function if there's no previous instructor turn

    # Create columns for the overview
    cols = st.columns(col_layout)
    col_time, col_i, col_act1, col_act2 = cols  # Split into two action columns
    # col_time, col_i, col_act, col_actvis = st.columns(col_layout)
    # Adding titles for each column
    col_act1.markdown("### True Answer")
    col_act2.markdown("### Model Prediction")
    col_time.markdown("<br><br><br>", unsafe_allow_html=True)  # Add this to align the first item with the titles
    col_i.markdown("<br><br><br>", unsafe_allow_html=True)

    for i in range(previous_instructor_turn_idx, turn + 1):
        d = data[i]
        
        if i > 0 and show_advanced_info:
            # Use html to add a horizontal line with minimal gap
            st.markdown(
                "<hr style='margin-top: 0.1rem; margin-bottom: 0.1rem;'/>",
                unsafe_allow_html=True,
            )
      
        if i == 0:
            col_time.markdown("<br><br><br>", unsafe_allow_html=True)  # Add this to align the first item with the titles
            col_i.markdown("<br><br><br>", unsafe_allow_html=True) 
        
        
        secs_from_start = d["timestamp"] - data[0]["timestamp"] #data
        # `secs_from_start` is a float including ms, display in MM:SS.mm format
        col_time.markdown(
            f"**{datetime.utcfromtimestamp(secs_from_start).strftime('%M:%S')}**"
        )
        
        if not st.session_state.get("enable_html_download", True):
            col_i.markdown(f"**#{i}**")
        
        elif d["type"] == "browser" and (page_filename := d["state"]["page"]):
            page_path = f"{basedir}/pages/{page_filename}"

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
        predicted_action = pred_mapping[str((demo_name, turn))]
        
        if i == turn:
            screenshot_path = execute_action(predicted_action)
            if screenshot_path:
            
                imgg = Image.open(screenshot_path)
                col_act2.image(imgg, caption="Screenshot after action")
            else: 
                col_act2.image(img)
        else:
            col_act2.image(img)
        if img:
            col_act1.image(img)
        
        # col_act.markdown(action_str)
        col_act1.markdown(action_str)
        if i == turn:
            
            # key = str((demo_name, turn))
            predicted_action = pred_mapping[str((demo_name, turn))]
            action_type = predicted_action[1][0]
            action_class = predicted_action[3][0]
            action_uid = predicted_action[2]
            xpath = predicted_action[3][1]
            st.write(f"New Predicted Action: {action_type} {action_class}")
            
            pred_action = f"**{action_type}**(uid={action_uid})"
            # st.write(f"Predicted Action: {pred_action}")
            col_act2.markdown(pred_action)
            # col_act2.markdown(action_str)
        else:
            col_act2.markdown(action_str)#predicted_action_str

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

    replay_file = replay_file.replace(".json", "")
    
    if not Path(basedir).joinpath('metadata.json').exists():
        st.error(f"Metadata file not found at {basedir}/metadata.json. This is likely an issue with Huggingface Spaces. Try cloning this repo and running locally.")
        st.stop()
    
    metadata = load_json_no_cache(basedir, "metadata")

    # Read in the JSON data
    replay_dict = load_json_no_cache(basedir, replay_file)
    #get first element of replay_dict
    

    form = load_json_no_cache(basedir, "form")
    
    if replay_dict is None:
        st.error(f"Replay file not found at {basedir}/{replay_file}. This is likely an issue with Huggingface Spaces. Try cloning this repo and running locally.")
        st.stop()
    
    if form is None:
        st.error(f"Form file not found at {basedir}/form.json. This is likely an issue with Huggingface Spaces. Try cloning this repo and running locally.")
        st.stop()


    st.sidebar.markdown("---")
    
    processed_meta_path = Path(basedir).joinpath('processed_metadata.json')
    start_frame = 'file not found'
    
    if processed_meta_path.exists():
        with open(processed_meta_path) as f:
            processed_meta = json.load(f)
        start_frame = processed_meta.get('start_frame', 'info not in file')
    
    data = replay_dict["data"]
    return data


def run():
    
    # print("splits", datasets)
    setup_browser()
    try:
        model_choice = st.sidebar.selectbox(
            "Choose your Llama model:",
            ["1.3B", "2.7B"],
            index=0  # Default selection
        )
        # st.sidebar.write(f"You have selected the model: {model_choice}")
        demonstration_dir = "./wl_data/demonstrations"
        
        demo_names = os.listdir(demonstration_dir)

        dataset = st.sidebar.selectbox("Select Dataset", list(unique_data_dict.keys()))
        if dataset:
            demos = unique_data_dict[dataset]
            demo_name = st.sidebar.selectbox("Select Demo", list(demos.keys()))

            # Dropdown to select the turn number based on the selected demo
            if demo_name:
                turns = demos[demo_name]
                selected_turn = st.sidebar.selectbox("Select Turn Number", sorted(turns))

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

                    show_overview(data, model_name=model_choice, recording_name=recording_name,dataset=dataset, demo_name=demo_name, turn=selected_turn, basedir=basedir)
    finally:
        shutdown_browser()
    

if __name__ == "__main__":
    # init_model()
    install_playwright()
    setup_datasets()
    # st.write("unique tuples", unique_data_dict)
    # st.set_page_config(layout="wide")
    run()