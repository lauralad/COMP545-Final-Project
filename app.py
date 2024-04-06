from flask import Flask, render_template, request, jsonify
import asyncio
from playwright.async_api import async_playwright
import re

app = Flask(__name__)

actions_list = ["say(speaker='navigator', utterance='Sure.') load(url='http://encyclopedia.com/') click(uid='rcLink') click(uid='')</s><s>[INST]"
               ]
# Global variables for the browser and page instances
browser_instance = None
page_instance = None


async def get_browser_page():
    global browser_instance, page_instance
    # Ensure the playwright and browser are started.
    if browser_instance is None:
        playwright = await async_playwright().start()
        browser_instance = await playwright.chromium.launch(headless=False)
    if page_instance is None or page_instance.is_closed():
        page_instance = await browser_instance.new_page()
    return page_instance, browser_instance

async def close_browser():
    global browser_instance
    if browser_instance is not None:
        await browser_instance.close()
        browser_instance = None

async def perform_web_action(action_type, params):
    global page_instance, browser_instance
    # if not page_instance:
    #     page_instance = await get_browser_page()
    page_instance, browser_instance = await get_browser_page()
    if action_type == "click":
        xpath = f"#{params['uid']}"
        print(f"Attempting to locate element with XPath: {xpath}")
        await page_instance.wait_for_selector(xpath, state="visible")
        print("Element found. Attempting to click.")
        await page_instance.click(xpath)
        print("Click action completed successfully.")

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

# async def perform_web_action(action_type, params):
#     global page_instance
#     page = await get_browser_page()

#     try:
#         if action_type == "click":
#             xpath = f"#{params['uid']}"
#             print(f"Attempting to locate element with XPath: {xpath}")
#             await page.wait_for_selector(xpath, state="visible")
#             print("Element found. Attempting to click.")
#             await page.click(xpath)
#             print("Click action completed successfully.")

#         elif action_type == "load":
#             await page.goto(params['url'])

#         # Add other actions as elif blocks
#         print(f"Action {action_type} completed successfully.")

#     except Exception as e:
#         print(f"Error performing {action_type}: {e}")
#         # Optionally re-initialize the browser if certain errors are encountered
#         await close_browser()


def parse_model_output(output):
    actions = re.findall(r"(\w+)\(([^)]+)\)", output)
    parsed_actions = []
    for action in actions:
        action_type, params_str = action
        # Matching key=value pairs, allowing for values in double quotes that may contain spaces or special characters
        params = dict(re.findall(r'(\w+)=["\']?([^"\']+)["\']?', params_str))
        parsed_actions.append((action_type, params))
    return parsed_actions

async def search_and_highlight_info(query):
    # Open Wikipedia
    # page_instance, browser_instance = await get_browser_page()
    # page = await browser.new_page()
    page_instance, browser_instance = await get_browser_page()
    await page_instance.goto('https://en.wikipedia.org/wiki/Main_Page')
    await asyncio.sleep(7)
    
    # Add 'say' action to the list of actions
    actions = [("say", {"speaker": "navigator", "utterance": "Sure."})]
    
    # Perform search
    await page_instance.fill('input[name="search"]', query)
    await page_instance.press('input[name="search"]', 'Enter')
    
    # Wait for search results
    await page_instance.wait_for_selector('.searchresult')
    
    # Click on the search result for McGill University
    await page_instance.click('a[href*="McGill_University"]')
    
    # Wait for page to load
    await page_instance.wait_for_load_state('networkidle')
    
    # Introduce a 30-second delay
    await asyncio.sleep(7)
    
    # Extract and highlight the information about McGill's founding date
    founding_date_element = await page_instance.query_selector('th:has-text("Established") + td')
    if founding_date_element:
        # Highlight the founding date text
        await founding_date_element.evaluate('(element) => { element.style.backgroundColor = "yellow"; }')
        # Get the text content of the founding date element
        founding_date = await founding_date_element.text_content()
        return founding_date
@app.route("/")
async def index():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
async def get_bot_response_route():
    user_text = request.args.get('msg')
    # Implement your conditions for handling user text
    if user_text.lower() in ["hello", "hi", "good afternoon"]:
        response = "Hi there! How can I help you?"
    else:
        # This is where you'd use your NLP model or some logic to determine the action
        # For demonstration, this is hardcoded to simulate an action
        # model_output = actions_list.pop(0)
        # actions = parse_model_output(model_output)
        # actions = []

        # response = ""
        # for action_type, params in actions:
        #     print("taking action:", action_type, params)
        #     await perform_web_action(action_type, params)
        #     if action_type == "say":
        #         response = params.get("utterance", "")
        #     await asyncio.sleep(30)
                # Assume the user wants to perform a search
        response = "Sure, let me look that up for you."
        # Call the search function
        founding_date = await search_and_highlight_info("McGill")
        if founding_date:
            response = f"The founding date of McGill University is {founding_date}."
    
    return jsonify(response)

    # return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
