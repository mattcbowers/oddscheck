from flask import render_template
from flask import request
from flaskexample import app
from .utils import generate_output
from .utils import generate_maybe
from .utils import get_probability
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
    # Get Inputs
    resource = request.args.get('resource', '')
    grade = request.args.get('grade', '')
    prefix = request.args.get('prefix', '')
    state = request.args.get('state', '')
    poverty = request.args.get('poverty', '')
    query = request.args.get('query', '')
    # Get the outputs
    res1, res2 = get_probability(resource, grade, prefix, state, poverty, query)
    # Render the output page
    return render_template(
        'go.html',
        res1 = res1,
        res2 = res2,
    )
