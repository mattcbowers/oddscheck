from flask import render_template
from flask import request
from flaskexample import app
from .utils import generate_output
from .utils import generate_maybe
from .utils import get_probability
from .utils import get_prob_df
#import flaskexample.utils
# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
import pandas as pd
# import psycopg2


@app.route('/', methods=["GET", "POST"])
@app.route('/index', methods=["GET", "POST"])
def index():
    df = ''
    search_params = {}
    if request.method == "POST":
        # Get Inputs
        resource = request.form.get('resource', '')
        grade = request.form.get('grade', '')
        prefix = request.form.get('prefix', '')
        state = request.form.get('state', '')
        poverty = request.form.get('poverty', '')
        query = request.form.get('query', '')
        search_params = {
            "query": query,
            "grade": grade,
            "prefix": prefix,
            "poverty": poverty,
            "state": state,
            "resource": resource
        }
        # Get the outputs
        df = get_prob_df(resource, grade, prefix, state, poverty, query)
        df = df.to_html(index=True)
        #df = df.replace('<table','<font size="20" face="Courier New" ><table')
        #df = df.replace('</table>','</table></font>')
        #print(df)
    return render_template(
        'master.html',
        res_df = df,
        search_params = search_params
    )
