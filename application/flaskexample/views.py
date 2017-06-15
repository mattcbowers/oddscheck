from flask import render_template
from flask import request
from flaskexample import app
from .utils import generate_output
#import flaskexample.utils
# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
# import pandas as pd
# import psycopg2


@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')


@app.route('/go')
def go():
    query = request.args.get('query', '')
    query2 = request.args.get('query2', '')
    # out_text = query.upper()
    out_text = generate_output(query)
    return render_template(
        'go.html',
        out_text = out_text,
        out_text2 = 'hello world',
    )
