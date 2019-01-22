from flask import Flask, render_template, jsonify
from flask_cors import CORS, cross_origin
from flask_wtf.csrf import CSRFProtect

# csrf = CSRFProtect()
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# @csrf.exempt
@app.route('/_get_data/', methods=['POST'])
def _get_data():
	myList = ['Element1', 'Element2', 'Element3']
	return jsonify({'data' : myList})

if __name__ == "__main__":
	app.run(debug=True)
