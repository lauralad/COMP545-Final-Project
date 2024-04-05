from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import pipeline
import weblinx as wl
import torch
import re


def extract_content_template(text):

    start_html = re.escape("<s>[INST] <<SYS>>")
    end_html = re.escape("You")
    start_utterances = re.escape("The user's first and last 4 utterances are:")
    end_utterances = re.escape(";")
    start_viewport = re.escape("Viewport size:")
    end_viewport = re.escape("Only the last")
    start_candidates = re.escape("this turn:")
    end_candidates = re.escape("<</SYS>>")
    start_action_history = re.escape("[/INST]")
    end_action_history = re.escape("Please select")
    
    
    html_pattern = fr"{start_html}(.*?){end_html}"
    utterances_pattern = fr"{start_utterances}(.*?){end_utterances}"
    viewport_pattern = fr"{start_viewport}(.*?){end_viewport}"
    candidates_pattern = fr"{start_candidates}(.*?){end_candidates}"
    action_history_pattern = fr"{start_action_history}(.*?){end_action_history}"
    
    # Search for the pattern
    clean_html = re.search(html_pattern, text, re.DOTALL).group(1).strip()
    utterances = re.search(utterances_pattern, text, re.DOTALL).group(1).strip() + " ;"
    viewport = re.search(viewport_pattern, text, re.DOTALL).group(1).strip()
    candidates = re.search(candidates_pattern, text, re.DOTALL).group(1).strip()
    action_history = re.search(action_history_pattern, text, re.DOTALL).group(1).strip()
    
    return clean_html, utterances, viewport, candidates, action_history

def extract_pred(text):
    pattern_inst = re.compile(r"(.*?\[INST\])", re.DOTALL)
    match_inst = pattern_inst.search(text)
    
    if match_inst:
        return match_inst.group(1)
    else:
        # If "[INST]" is not found, find the first ")" and capture everything up to and including it
        pattern_parenthesis = re.compile(r"(.*?\))", re.DOTALL)
        match_parenthesis = pattern_parenthesis.search(text)
        
        if match_parenthesis:
            return match_parenthesis.group(1)
        else:
            return "Neither [INST] nor closing parenthesis found."

def run_and_extract_pred(text):
    with open('templates/llama.txt') as f:
        template = f.read()
    turn = text
    turn_text = template.format(**turn)
    #adjust device either to -1 for cpu, or "device" if previously defined
    action_model = pipeline(
        model="McGill-NLP/Sheared-LLaMA-2.7B-weblinx", device=-1, torch_dtype='auto'
    )
    out = action_model(turn_text, return_full_text=False, max_new_tokens=64, truncation=True)
    pred = out[0]['generated_text']
    final_pred = extract_pred(pred)
    
    return final_pred
        
