from flask import render_template
from flask import request
from flaskexample import app
from .utils import generate_output
from .utils import generate_maybe
from .utils import get_probability
#import flaskexample.utils
# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
import pandas as pd
# import psycopg2


@app.route('/', methods=["GET", "POST"])
@app.route('/index', methods=["GET", "POST"])
def index():
    res1 = ""
    res2 = ""
    df = None
    if request.method == "POST":
        # Get Inputs
        resource = request.form.get('resource', '')
        grade = request.form.get('grade', '')
        prefix = request.form.get('prefix', '')
        state = request.form.get('state', '')
        poverty = request.form.get('poverty', '')
        query = request.form.get('query', '')
        print(resource, grade, prefix, state, poverty, query)
        # Get the outputs
        res1, res2 = get_probability(resource, 
            grade, prefix, state, poverty, query)
        # Render the output page
        df = pd.DataFrame({
          "Price": range(500, 5000, 500),
          "Confidence": range(10, 100, 10)
        })
        df = df.to_html(index=False)
    return render_template(
        'master.html',
        res1 = res1,
        res2 = res2,
        res_df = df
    )
