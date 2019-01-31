from flask import Flask, render_template, jsonify
from flask_cors import CORS, cross_origin
# from flask_wtf.csrf import CSRFProtect
# from flask_sslify import SSLify


# csrf = CSRFProtect()
app = Flask(__name__)
# sslify = SSLify(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# @csrf.exempt
@app.route('/_get_data/', methods=['POST'])

def _get_data():
	myList = ['Element1', 'Element2', 'Element3']
	#return jsonify({'data' : myList})
	return jsonify({'data': render_template('response.html', myList=myList)})

if __name__ == "__main__":
	app.run(debug=True, threaded=True)

#generate SSL Certificate
#OpenSSL for Flask