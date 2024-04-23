from quart import Quart, render_template, request, jsonify
import asyncio
from playwright.async_api import async_playwright
import re

app = Quart(__name__)

# actions_list = [
#     "say(speaker='navigator', utterance='Sure.') load(url='http://encyclopedia.com/') click(uid='rcLink')</s><s>[INST]",
#     "say(speaker='navigator', utterance='Sure.') click(href='/medicine')</s><s>[INST]"
# ]

actions_list = ["say(speaker='navigator', utterance='Sure.') load(url='https://google.com/')</s><s>[INST]",
                "say(speaker='navigator', utterance='Alright.') click(uid='#APjFqb') text_input(uid='APjFqb', text='wikipedia')</s><s>[INST]",
                "say(speaker='navigator', utterance='Ok.') click(href='https://en.wikipedia.org/wiki/Main_Page')</s><s>[INST]",
                "say(speaker='navigator', utterance='Ok.') click(name='search') text_input(name='search', text='McGill')</s><s>[INST]",
                "say(speaker='navigator', utterance='Sure.') click(href=/wiki/McGill_University')</s><s>[INST]",
                "say(speaker='navigator', utterance='McGill University was founded in 1821.') find(text='Founded in')</s><s>[INST]"
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

async def close_browser():
    global browser_instance
    if browser_instance is not None:
        await browser_instance.close()
        browser_instance = None

async def perform_web_action(action_type, params):
    global page_instance, browser_instance
    
    if action_type == "click":
        if 'uid' in params and params['uid']:
            xpath = f"{params['uid']}" #add # for id after video
            print(f"Attempting to locate element with ID: {xpath}")
            await page_instance.wait_for_selector(xpath, state="visible")
            print("Element found. Attempting to click.")
            await page_instance.click(xpath)
            
        # If params specify a 'href', use it as an href selector
        elif 'href' in params and params['href']:
            href_selector = f"a[href='{params['href']}']"
            print(f"Attempting to locate element with href: {params['href']}")
            await page_instance.wait_for_selector(href_selector, state="visible")
            print("Element found. Attempting to click.")
            await page_instance.click(href_selector)
        elif 'name' in params and params['name']:
            name_selector = f"input[name='{params['name']}']"
            print(f"Attempting to locate element with name: {params['name']}")
            await page_instance.wait_for_selector(name_selector, state="visible")
            print("Element found. Attempting to click.")
            await page_instance.click(name_selector)
        
        print("Click action completed successfully.")

    elif action_type == "load":
        await page_instance.goto(params['url'])

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
        if 'name' in params and params['name']:
            xpath = f"input[name='{params['name']}']"
        else:
            xpath = f"#{params['uid']}"
        await page_instance.locator(xpath).click(force=True)
        print("clicked!")
        # Wait for the input element to become visible
        # await page_instance.wait_for_selector(input_selector, state="visible", timeout=5000)
        
        await page_instance.fill(xpath, params['text'])
        await page_instance.keyboard.press('Enter')

        print(f"Filled the input field ({xpath}) with text: {params['text']}")


    elif action_type == "change":
        # Assuming 'change' refers to changing the value of an input element
        input_xpath = f"//*[@id='{params['uid']}']"
        await page_instance.fill(input_xpath, params['value'])

    elif action_type == "find":
        # New logic to highlight text on the page
        if 'text' in params and params['text'] == "Founded in":
            script = """
            const paragraphs = Array.from(document.querySelectorAll('p'));
            paragraphs.forEach(paragraph => {
                const regex = /(Founded in )(\d{4})/gi;
                paragraph.innerHTML = paragraph.innerHTML.replace(regex, (match, prefix, year) => {
                    return prefix + `<span style="background-color: yellow; font-weight: bold;">${year}</span>`;
                });
            });
            """
            await page_instance.evaluate(script)
            print("Year highlighting completed.")


def parse_model_output(output):
    actions = re.findall(r"(\w+)\(([^)]+)\)", output)
    parsed_actions = []
    for action in actions:
        action_type, params_str = action
        params = dict(re.findall(r'(\w+)=["\']?([^"\']+)["\']?', params_str))
        parsed_actions.append((action_type, params))
    return parsed_actions

@app.route("/")
async def index():
    return await render_template("index.html")

@app.route("/get", methods=["GET"])
async def get_bot_response_route():
    global page_instance
    user_text = request.args.get('msg')
    await get_browser_page()
    
    if user_text.lower() in ["hello", "hi", "good afternoon"]:
        response = "Hi there! How can I help you?"
    else:
        model_output = actions_list.pop(0) if actions_list else "say(speaker='navigator', utterance='I am sorry, I am not sure how to respond to that.')"
        actions = parse_model_output(model_output)

        response = ""
        for action_type, params in actions:
            print("taking action:", action_type, params)
            await perform_web_action(action_type, params)
            if action_type == "say":
                response += params.get("utterance", "") + " "
    
    return jsonify(response)

# @app.before_serving
# async def startup():
#     await get_browser_page()

@app.after_serving
async def cleanup():
    await close_browser()

if __name__ == "__main__":
    app.run(debug=True)

