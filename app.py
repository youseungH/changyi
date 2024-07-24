from flask import Flask, request
from flask_restful import Api, Resource, reqparse
from mylib.classification  import Jaenan_moonja

data = None
app = Flask(__name__)
message = Jaenan_moonja(data)

@app.route('/subject', methods=['POST'])
def find_subject(text) :
    data = text 
    message.find_subject()
    message.sebject_of_message()


@app.route('/importance', methods=['POST'])
def find_subject(text) :
    data = text 
    message.is_important()

if __name__ == '__main__':
    app.run()
