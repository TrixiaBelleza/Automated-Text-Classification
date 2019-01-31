from flask import Flask, render_template, request, jsonify, json
from flask_cors import CORS, cross_origin
# from flask_wtf.csrf import CSRFProtect
# from flask_sslify import SSLify


# csrf = CSRFProtect()
app = Flask(__name__)
# sslify = SSLify(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# @csrf.exempt
@app.route('/_get_data/', methods=['GET'])

def _get_data():
	myList = ['Element1', 'Element2', 'Element3']
	#return jsonify({'data' : myList})
	return jsonify({'data': render_template('response.html', myList=myList)})

@app.route('/_fetch_input/', methods=['POST'])

def _fetch_input():
	print(request.get_data().decode("utf-8"))
	return json.dumps({'status':'OK'});


if __name__ == "__main__":
	app.run(debug=True, threaded=True)

#generate SSL Certificate
#OpenSSL for Flask