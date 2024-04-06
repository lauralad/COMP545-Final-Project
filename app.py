from flask import Flask, render_template, request, jsonify
import asyncio
from playwright.async_api import async_playwright
import re

app = Flask(__name__)

actions_list = ["say(speaker='navigator', utterance='Sure.') load(url='http://encyclopedia.com/') click(uid='rcLink')</s><s>[INST]", 
                "say(speaker='navigator', utterance='Alright.') click(uid='rcLink')</s><s>[INST]"
               ]
# Global variables for the browser and page instances
browser_instance = None
page_instance = None


async def get_browser_page():
    global global_browser, global_page
    if global_browser is None:
        playwright = await async_playwright().start()
        global_browser = await playwright.chromium.launch(headless=False)
    if global_page is None:
        global_page = await global_browser.new_page()
    return global_page

async def close_browser():
    global browser_instance
    if browser_instance is not None:
        await browser_instance.close()
        browser_instance = None

# async def handle_cookie_popup(page):
#     # Replace 'button.close' with the actual selector for the popup's close button
#     close_button_selector = 'button.close'  
#     try:
#         # Wait for the popup to appear. Adjust the timeout as necessary.
#         await page.wait_for_selector(close_button_selector, state='visible', timeout=5000)
#         await page.click(close_button_selector)
#         print("Cookie popup handled.")
#     except Exception as e:
#         print("No cookie popup appeared, or there was an issue closing it.", e)

async def perform_web_action(action_type, params):
    global page_instance
    if not page_instance:
        page_instance = await get_browser_page()


    if action_type == "click":
        try:
            # Using XPath to select the element by its ID, accommodating IDs that start with digits
            # xpath = f"//*[@id='{params['uid']}']"
            xpath = f"#{params['uid']}"
            await page_instance.goto("http://encyclopedia.com/")
            await page_instance.wait_for_selector(xpath, state="visible", timeout=6000)  # Ensure the element is ready
            await page_instance.locator(xpath).click(force=True)  # Use force=True to click regardless of being covered
            print("clicked!")
        except Exception as e:
            print(f"An error occurred while clicking: {e}")

    elif action_type == "load":
        await page_instance.goto(params['url'])
        # body_html = await page_instance.evaluate("() => document.body.innerHTML")

        # # Save to a text file
        # with open('templates/body_content.txt', 'w', encoding='utf-8') as file:
        #     file.write(body_html)

        # full_html = await page_instance.evaluate("() => document.documentElement.outerHTML")

        # # Save to a text file
        # with open('templates/full_document_content.txt', 'w', encoding='utf-8') as file:
        #     file.write(full_html)
        # await handle_cookie_popup(page_instance)
        # # If needed, wait a bit after closing the popup to ensure the page_instance stabilizes
        # await page_instance.wait_for_timeout(1000)
        # page_instance_content = await page_instance.content()
        # print("page_instance content loaded:", page_instance_content)

    elif action_type == "scroll":
        # Note: Playwright does not directly support scroll to x, y. You may need to execute custom JavaScript.
        await page_instance.evaluate(f"window.scrollTo({params['x']}, {params['y']})")

    elif action_type == "submit":
        # Assuming the uid is the form element or a submit button within the form
        form_xpath = f"//*[@id='{params['uid']}']"
        await page_instance.wait_for_selector(form_xpath, state="visible", timeout=60000)
        # Triggering form submission
        await page_instance.evaluate(f"document.querySelector('{form_xpath}').submit();")

    elif action_type == "text_input":
        # input_selector = "input.searchbox.form-search.form-input"
        # print("in text input if")
        xpath = f"#{params['uid']}"
        await page_instance.locator(xpath).click(force=True)
        print("clicked!")
        # Wait for the input element to become visible
        # await page_instance.wait_for_selector(input_selector, state="visible", timeout=5000)
        
        await page_instance.fill(xpath, params['text'])

        print(f"Filled the input field ({xpath}) with text: {params['text']}")


    elif action_type == "change":
        # Assuming 'change' refers to changing the value of an input element
        input_xpath = f"//*[@id='{params['uid']}']"
        await page_instance.fill(input_xpath, params['value'])


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
        # Example model output, replace this with your actual model interaction text_input(text='biotechnology', uid='67e2a5fb-8b1d-41a0')
        model_output = actions_list.pop(0)
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
                print("taking action:", action_type, params)
                await perform_web_action(action_type, params)
        
        # Return the 'say' action utterance, or a default message if none is found
        return jsonify(say_utterance if say_utterance else "Action is being processed.")



if __name__ == "__main__":
    app.run(debug=True)
