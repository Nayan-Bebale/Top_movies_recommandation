from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired()])
    submit = SubmitField('Register')


class Edit(FlaskForm):
    your_rating = StringField('Your Rating Out of 10 e.g. 7.3', validators=[DataRequired()])
    your_review = StringField('Your Review', validators=[DataRequired()])
    done = SubmitField('Done')


class AddMovies(FlaskForm):
    movie_title = StringField('Movie Title', validators=[DataRequired()])
    done = SubmitField('Add Movie')

class SearchMovies(FlaskForm):
    movie_title = StringField('Movie Title', validators=[DataRequired()])
    done = SubmitField('Search Movie')

