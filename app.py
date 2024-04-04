from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    # Return to serving your HTML template
    return render_template("index.html")

@app.route("/get")
def get_bot_response_route():
    # Dummy implementation of your AJAX endpoint
    user_text = request.args.get('msg')
    if user_text.lower() in ["hello", "hi"]:
        response = "Hi there! How can I help you?"
    else:
        response = "I'm not sure how to respond to that. Can you try asking something else?"
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
