from flask import Flask, render_template, request, jsonify
import asyncio
from playwright.async_api import async_playwright
import re

app = Flask(__name__)


# Global variables for the browser and page instances
browser_instance = None
page_instance = None

async def get_browser_page():
    global browser_instance, page_instance
    if browser_instance is None:
        playwright = await async_playwright().start()
        browser_instance = await playwright.chromium.launch(headless=False)  # Set headless=True in production
        page_instance = await browser_instance.new_page()
    return page_instance

async def close_browser():
    global browser_instance
    if browser_instance is not None:
        await browser_instance.close()
        browser_instance = None

async def perform_web_action(action_type, params):
    page = await get_browser_page()

    if action_type == "click":
        selector = f"[id='{params['uid']}']"
        await page.click(selector)
        # await page.click(params['uid'])

    elif action_type == "load":
        await page.goto(params['url'])
        # page_content = await page.content()
        # print("Page content loaded:", page_content)

    elif action_type == "scroll":
        # Note: Playwright does not directly support scroll to x, y. You may need to execute custom JavaScript.
        await page.evaluate(f"window.scrollTo({params['x']}, {params['y']})")

    elif action_type == "submit":
        # Assuming the uid is the form element or a submit button within the form
        form_xpath = f"//*[@id='{params['uid']}']"
        await page.wait_for_selector(form_xpath, state="visible", timeout=60000)
        # Triggering form submission
        await page.evaluate(f"document.querySelector('{form_xpath}').submit();")

    elif action_type == "text_input":
        input_xpath = f"//*[@id='{params['uid']}']"
        await page.fill(input_xpath, params['text'])
        
    elif action_type == "change":
        # Assuming 'change' refers to changing the value of an input element
        input_xpath = f"//*[@id='{params['uid']}']"
        await page.fill(input_xpath, params['value'])


def parse_model_output(output):
    actions = re.findall(r"(\w+)\(([^)]+)\)", output)
    parsed_actions = []
    for action in actions:
        action_type, params_str = action
        # Matching key=value pairs, allowing for values in double quotes that may contain spaces or special characters
        params = dict(re.findall(r'(\w+)=["\']?([^"\']+)["\']?', params_str))
        parsed_actions.append((action_type, params))
    return parsed_actions


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
async def get_bot_response_route():
    user_text = request.args.get('msg')
    if user_text.lower() in ["hello", "hi"]:
        return jsonify("Hi there! How can I help you?")
    else:
        # Example model output, replace this with your actual model interaction
        model_output = "say(speaker='navigator', utterance='Sure.') load(url='https://www.encyclopedia.com/') click(uid='67e2a5fb-8b1d-41a0')</s><s>[INST]"
        actions = parse_model_output(model_output)
        print("actions:", actions)
        # Initialize a variable to hold any 'say' action utterances
        say_utterance = ""
        for action_type, params in actions:
            if action_type == "say":
                # Directly capture the 'say' action's utterance
                say_utterance = params.get("utterance", "")
            else:
                # Perform other web actions silently
                await perform_web_action(action_type, params)
        
        # Return the 'say' action utterance, or a default message if none is found
        return jsonify(say_utterance if say_utterance else "Action is being processed.")



if __name__ == "__main__":
    app.run(debug=True)
