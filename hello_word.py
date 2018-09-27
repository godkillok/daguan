from flask import Flask,request,abort
import json
app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello():
    return 'Hello World'

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % username

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id

@app.route('/ana', methods=['GET', 'POST'])
def call_analysis():
    print ("Hello, World!")
    if not request.json:
      abort(400)

    data = request.get_data()
    data=str(data,encoding='utf-8')
    print(data)
    data=json.loads(data)
    for (k,v) in data.items():
        data[k]=v*10
    data=json.dumps(data)
    return data


if __name__ == '__main__':
    app.run()