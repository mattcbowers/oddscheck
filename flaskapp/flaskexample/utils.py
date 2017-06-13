def generate_output(query):
    import pandas as pd
    import pickle
    from sklearn.linear_model import LogisticRegression
    cost = float(query)
    # create data frame from query
    new_data = pd.DataFrame({
        'total_price_excluding_optional_support': [cost],
        'students_reached': [15]
    })
    filename_model = 'flaskexample/models/mod_logit_simple.pkl'
    filename_scaler = 'flaskexample/models/mod_logit_simple_scaler.pkl'
    mod_logit = pickle.load(open(filename_model, 'rb'))
    scaler = pickle.load(open(filename_scaler, 'rb'))
    prob = mod_logit.predict_proba(scaler.transform(new_data))[0][1]
    prob_str = str(int(round(100 * prob))) + '%'
    out_str = 'Proposed cost: $' + query + ' Funding probability: ' + prob_str
    return(out_str)
