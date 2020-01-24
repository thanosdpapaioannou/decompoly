from flask import Flask, render_template, flash, request, Markup
from wtforms import Form, StringField, validators
from src.opt import get_sos
from sympy import poly, latex, sympify, nan
from src.poly import get_latex_from_poly, is_polynomial

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'SjdnUends821Jsdlkvxh391ksdODnejdDw'


class ReusableForm(Form):
    polynomial = StringField('Polynomial', validators=[validators.DataRequired()])


@app.route('/', methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
    if request.method == 'POST':
        _input = request.form['polynomial']
        if form.validate() and is_polynomial(_input):
            _polynomial = poly(_input)
            msg, sos = get_sos(_polynomial)
            if sos == nan:
                flash(Markup(f'<strong>Input</strong>: \({latex(sympify(_input))}\)'))
                flash(Markup(f'<strong>Result</strong>: {msg}'))
            else:
                flash(Markup(f'<strong>Result</strong>: \({latex(sympify(_input))} = {get_latex_from_poly(sos)}\)'))
        else:
            flash('Error: Non-constant polynomial required')
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run()
