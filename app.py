from flask import Flask, render_template, flash, request
from wtforms import Form, StringField, validators
from decompoly import get_sos
from sympy import poly, latex, sympify, nan

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'SjdnUends821Jsdlkvxh391ksdODnejdDw'


class ReusableForm(Form):
    polynomial = StringField('Polynomial', validators=[validators.DataRequired()])


def get_latex_from_poly(polynomial):
    _latex_string = latex(polynomial, mode='plain')
    _poly_str = _latex_string.split(',')[0].replace('\operatorname{Poly}{\left(', '').strip()
    return _poly_str


def is_input_polynomial(input):
    try:
        _polynomial = poly(input)
    except:
        return False
    return True


@app.route('/', methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
    if request.method == 'POST':
        _input = request.form['polynomial']
        if form.validate() and is_input_polynomial(_input):
            _polynomial = poly(_input)
            msg, sos = get_sos(_polynomial)
            if sos == nan:
                flash(msg)
            else:
                flash(f'\({latex(sympify(_input))} = {get_latex_from_poly(sos)}\)')
        else:
            flash('Error: Non-constant polynomial required')
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run()
