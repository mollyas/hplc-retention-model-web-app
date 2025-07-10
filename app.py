# Imports
from flask import Flask, render_template, request, jsonify
from NKGradModel import get_nkGrad_params
import numpy as np
from NKIsoModel import find_k, get_nkIso_params

# My App
app = Flask(__name__)

# Homepage
@app.route("/")

@app.route('/<page_name>')
def index(page_name='home'):  # Default to 'home' page if no page_name is provided
    valid_pages = ['home', 'about', 'model_fitter', 'terminology', 'other_works']

    if page_name not in valid_pages:
        return render_template('404.html'), 404

    return render_template(f'{page_name}.html', current_page=page_name)


@app.route('/process_model', methods=['POST'])
def process_model():
    data = request.get_json()
    chosenElution = data.get('chosenElution')
    chosenModel = data.get('chosenModel')

    # You can replace this with whatever logic you need
    if chosenElution == 'isocratic' and chosenModel == 'NK':
        html_block = render_template('models/isoNK.html')
    elif chosenElution == 'gradient' and chosenModel == 'NK':
        html_block = render_template('models/gradNK.html')
    elif chosenElution == 'isocratic' and chosenModel == 'LSS':
        html_block = render_template('models/isoLSS.html')
    elif chosenElution == 'gradient' and chosenModel == 'LSS':
        html_block = render_template('models/gradLSS.html')
    else:
        html_block = render_template('models/select.html')

    return jsonify({'result': html_block})


@app.route('/fit_iso_nk_model', methods=['POST'])
def fit_iso_nk_model():
    try:
        data = request.get_json()
        phi_values = np.array(data['phi'], dtype=float)
        retention_times = np.array(data['retention'], dtype=float)

        tm = float(data['tm'])
        tex = float(data['tex'])

        retention_factors = ((retention_times - tex) - (tm - tex)) / (tm - tex)


        if tm is None or tex is None:
            raise ValueError("Missing t_m or t_ex values")

        # Generate some params
        kw, S1, S2, perr = get_nkIso_params(phi_values, retention_times, tex, tm)

        # generate x, y data for the fit
        plot_phi_range = np.linspace(0, 100, 1000)
        plot_k_values = find_k(plot_phi_range, kw, S1, S2)


        return jsonify({
            'status': 'success',
            'x_fit': plot_phi_range.tolist(),
            'y_fit': plot_k_values.tolist(),
            'x_exp': phi_values.tolist(),
            'y_exp': retention_factors.tolist(),
            'kw': round(kw, 2),
            'kw_err': round(perr[0], 2),
            's1': round(S1, 2),
            's1_err': round(perr[1], 2),
            's2': round(S2, 2),
            's2_err': round(perr[2], 2)
        })

    except Exception as e:
        print(f"Error in fit_iso_nk_model: {e}")  # Log the error for debugging
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/fit_grad_nk_model', methods=['POST'])
def fit_grad_nk_model():
     try:
        data = request.get_json()
        phi_i = float(data['phi_i'])
        phi_f = float(data['phi_f'])
        tm = float(data['tm'])
        tex = float(data['tex'])
        td = float(data['td'])

        tg_values = np.array(data['tg'], dtype=float)
        retention_times = np.array(data['retention'], dtype=float)

        if phi_i is None or phi_f is None or td is None or tex is None or tm is None:
            raise ValueError("Missing values")

        # generate some params
        kw, S1, S2, perr = get_nkGrad_params(phi_i, phi_f, td, tm, tex, tg_values, retention_times)

        # generate x, y data for the fit
        plot_phi_range = np.linspace(0, 100, 1000)
        plot_k_values = find_k(plot_phi_range, kw, S1, S2)

        return jsonify({
           'status': 'success',
           'x_fit': plot_phi_range.tolist(),
           'y_fit': plot_k_values.tolist(),
           'kw': round(kw, 2),
           'kw_err': round(perr[0], 2),
           's1': round(S1, 2),
           's1_err': round(perr[1], 2),
           's2': round(S2, 2),
           's2_err': round(perr[2], 2)
         })

     except Exception as e:
         print(f"Error in fit_grad_nk_model: {e}")  # Log the error for debugging
         return jsonify({'status': 'error', 'message': str(e)})


# final test
if __name__ == "__main__":
    # debugger always on
    app.run(debug=True)
