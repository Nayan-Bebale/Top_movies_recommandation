import requests
import pickle
import pandas as pd
import logging
import os
import re
import asyncio
import aiohttp
import gzip
from dotenv import load_dotenv
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import UserMixin, LoginManager, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bootstrap import Bootstrap
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
from concurrent.futures import ThreadPoolExecutor
from fuzzywuzzy import process, fuzz
from tenacity import retry, stop_after_attempt, wait_fixed


load_dotenv()

db = SQLAlchemy()  # Create an instance of SQLAlchemy without initializing it with app.
migrate = Migrate()  # Create an instance of Migrate without initializing it with app.

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
bootstrap = Bootstrap(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movies.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)  # Initialize with the app only once
migrate.init_app(app, db)  # Initialize the migration manager with the app and db

API_KEY = os.getenv("API_KEY")

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Define common stopwords
stopwords = {"the", "a", "of", "and", "to", "in", "for", "with", "on", "at", "by", "an", "-", "as"}

# Genre mapping for dynamic URL construction
GENRE_IDS = {
    "action": 28,
    "adventure": 12,
    "comedy": 35,
    "drama": 18,
    "fantasy": 14,
    "horror": 27,
    "musicals": 10402,
    "mystery": 9648,
    "romance": 10749,
    "science_fiction": 878,
    "sports": 10770,
    "thriller": 53,
    "western": 37,
}

GENRE = {
        28: "Action" ,
        12: "Adventure",
        35: "Comedy",
        18: "Drama",
        14: "Fantasy",
        27: "Horror",
        10402: "Musicals",
        9648: "Mystery",
        10749: "Romance",
        878: "Science Fiction",
        10770: "Sports",
        53: "Thriller",
        37: "Western",
    }

# CREATE TABLE

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    movies = db.relationship('Movie', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(250), unique=True, nullable=False)
    year = db.Column(db.Integer, nullable=False)
    description = db.Column(db.String(500), nullable=False)
    rating = db.Column(db.Float, nullable=True)
    ranking = db.Column(db.Integer, nullable=True)
    review = db.Column(db.String(250), nullable=True)
    img_url = db.Column(db.String(250), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


with app.app_context():
    db.create_all()


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


@app.template_filter('truncate_words')
def truncate_words(s, num_words):
    words = s.split()
    if len(words) > num_words:
        return ' '.join(words[:num_words]) + '...'
    return s


app.jinja_env.filters['truncate_words'] = truncate_words


async def fetch_poster(session, movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US'
    async with session.get(url) as response:
        data = await response.json()
        return 'https://image.tmdb.org/t/p/w500/' + data['poster_path']


def clean_title(title):
    # Remove punctuation, but keep non-Latin characters
    title = re.sub(r'[^\w\s]', '', title)
    # Convert to lowercase
    title = title.lower()
    # Split title into words and filter out stopwords
    words = [word for word in title.split() if word not in stopwords]
    return " ".join(words)


def find_closest_match(movie_title, movie_list):
    cleaned_title = clean_title(movie_title)
    cleaned_movies = [clean_title(title) for title in movie_list]
    match = None
    score = 0
    if cleaned_movies:
        match, score = process.extractOne(cleaned_title, cleaned_movies, scorer=fuzz.partial_ratio)
    if match and score >= 70:  # Adjust the threshold as needed
        original_index = cleaned_movies.index(match)
        return movie_list[original_index]
    return None


async def get_recommendations(movie):
    movies_dict = pickle.load(open("./movie_dict.pkl", "rb"))
    # Load the compressed similarity file
    with gzip.open("similarity.pkl.gz", "rb") as f:
        similarity = pickle.load(f)

    movies = pd.DataFrame(movies_dict)

    if not movie:
        return "<h1>Movie title not provided.</h1>"

    # Tokenize the input movie title
    tokens = clean_title(movie).split()

    # Find potential matches
    potential_matches = []
    for token in tokens:
        potential_matches.extend(movies[movies['title'].str.contains(token, case=False)]['title'].tolist())

    # Use fuzzy matching to find the best match from potential matches
    best_match = find_closest_match(movie, potential_matches)

    if not best_match:
        return "<h1>Sorry, This movie is not found in the database.</h1>"

    movie_index = movies[movies['title'] == best_match].index[0]

    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = [movies.iloc[i[0]].title for i in movie_list]

    async with aiohttp.ClientSession() as session:
        tasks = [session.get(f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={title}") for title in
                 recommended_movies]
        search_responses = await asyncio.gather(*tasks)
        search_results = [await resp.json() for resp in search_responses]

        recommended_list = []
        tasks = []

        for result in search_results:
            if result['results']:
                movie_id = result['results'][0]['id']
                tasks.append(fetch_poster(session, movie_id))

        posters = await asyncio.gather(*tasks)

        for i, result in enumerate(search_results):
            if result['results']:
                data = result['results'][0]
                recommended_list.append({
                    "id": data['id'],
                    "title": data['original_title'],
                    "year": data['release_date'].split('-')[0],
                    "description": data['overview'],
                    "img_url": posters[i],
                })

    return recommended_list


def get_recommendations_sync(movie):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(get_recommendations(movie))


@app.route('/recommendation')
def recommend():
    movie = request.args.get('movie')

    if not movie:
        return "<h1>Movie title not provided.</h1>"

    with ThreadPoolExecutor() as executor:
        future = executor.submit(get_recommendations_sync, movie)
        recommended_list = future.result()

    if isinstance(recommended_list, str):  # This means an error message was returned
        return recommended_list

    return render_template("recommendation.html", movies=recommended_list)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# Retry logic for transient network issues
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def fetch_movie_data(session, url):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                return await response.json()
            else:
                logging.error(f"Error {response.status}: Failed to fetch data from {url}")
                return {}
    except aiohttp.ClientConnectorError as e:
        logging.error(f"Connection error while trying to access {url}: {e}")
        raise
    except asyncio.TimeoutError:
        logging.error(f"Timeout error while trying to access {url}")
        raise


async def get_homepage_data():
    import datetime
    current_year = datetime.datetime.now().year

    async with aiohttp.ClientSession() as session:
        # Common and genre-based URLs
        common_urls = {
            "top_3_movies": f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&language=en-US&sort_by=vote_average.desc&vote_count.gte=100&primary_release_year={current_year}&page=1",
            "latest_12_movies": f"https://api.themoviedb.org/3/movie/now_playing?api_key={API_KEY}&language=en-US&page=1",
            "upcoming_12_movies": f"https://api.themoviedb.org/3/movie/upcoming?api_key={API_KEY}&language=en-US&page=1",
            "director_choice": f"https://api.themoviedb.org/3/movie/top_rated?api_key={API_KEY}&language=en-US&page=1",
            "top_10_movies": f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&page=1",
            "best_movie_of_month": f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&region=US&page=1",

        }

        genre_urls = {
            f"trending_{genre}": f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_genres={genre_id}&sort_by=popularity.desc&page=1"
            for genre, genre_id in GENRE_IDS.items()
        }

        # Combine common and genre-based URLs
        urls = {**common_urls, **genre_urls}

        # Fetch data asynchronously
        tasks = [fetch_movie_data(session, url) for url in urls.values()]
        results = await asyncio.gather(*tasks)

        # Map results back to their keys
        results_dict = {key: result.get('results', []) for key, result in zip(urls.keys(), results)}

        # Limit results for specific categories
        return {
            "top_3_movies": results_dict["top_3_movies"][:3],
            "latest_12_movies": results_dict["latest_12_movies"][:12],
            "upcoming_12_movies": results_dict["upcoming_12_movies"][:12],
            "director_choice": results_dict["director_choice"][:6],
            "top_10_movies": results_dict["top_10_movies"][:10],
            "best_movie_of_month": results_dict["best_movie_of_month"][0] if results_dict[
                "best_movie_of_month"] else None,
            **{key: value[:4] for key, value in results_dict.items() if key.startswith("trending_")},
        }


@app.route("/")
def home():
    form = SearchMovies()
    # Fetch homepage data asynchronously
    homepage_data = asyncio.run(get_homepage_data())
    # Genre mapping for dynamic URL construction

    # Prepare context for rendering the template
    context = {
        'title': 'Home',
        'top_3_movies': homepage_data.get('top_3_movies', []),
        'latest_12_movies': homepage_data.get('latest_12_movies', []),
        'upcoming_12_movies': homepage_data.get('upcoming_12_movies', []),
        'director_choice': homepage_data.get('director_choice', []),
        'top_10_movies': homepage_data.get('top_10_movies', []),
        'best_movie_of_month': homepage_data.get('best_movie_of_month', {}),
        'genre_movies': {
            genre: homepage_data.get(f"trending_{genre}", [])
            for genre in [
                "action", "adventure", "comedy", "drama", "fantasy",
                "horror", "musicals", "mystery", "romance", "science_fiction",
                "sports", "thriller", "western"
            ]
        },
    }
    # print(homepage_data.get('best_movie_of_month'))

    # print(homepage_data.get('director_choice'))
    # Render the homepage with the prepared context
    return render_template("free_movie_zip/index.html", context=context, GENRE_IDS=GENRE, form=form)


@app.route("/about")
def about():
    return render_template("free_movie_zip/about.html", form = SearchMovies())

@app.route("/contact")
def contect():
    return render_template("free_movie_zip/contact.html", form = SearchMovies())

@app.route("/blog")
def blog():
    return render_template("free_movie_zip/blog.html", form = SearchMovies())

@app.route("/blog-details")
def blog_details():
    return render_template("free_movie_zip/blog_detail.html", form = SearchMovies())

@app.route("/services")
def services():
    return render_template("free_movie_zip/services.html", form = SearchMovies())

@app.route("/teams")
def teams():
    return render_template("free_movie_zip/team.html", form = SearchMovies())


@app.route("/search", methods=["GET", "POST"])
def search():
    form = SearchMovies()
    if form.validate_on_submit():
        movie_title = form.movie_title.data
        response = requests.get("https://api.themoviedb.org/3/search/movie",
                                params={"api_key": API_KEY, "query": movie_title})
        data = response.json()["results"]
        # print(data)
        return render_template("free_movie_zip/search.html", options=data, GENRE_IDS=GENRE, form=form)
    return redirect(url_for('home'))

@app.route("/data")
def data():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    movies = Movie.query.filter_by(user_id=current_user.id).order_by(Movie.id.desc()).all()
    return render_template("index.html", movies=movies)


@app.route("/edit", methods=["GET", "POST"])
def edit():
    form = Edit()
    movie_id = request.args.get("id")
    movie_selected = Movie.query.get_or_404(movie_id)

    if form.validate_on_submit():
        movie_selected.rating = float(form.your_rating.data)
        movie_selected.review = form.your_review.data
        db.session.commit()
        return redirect(url_for('home'))

    return render_template("edit.html", movie=movie_selected, form=form)


@app.route("/delete")
def delete():
    movie_id = request.args.get('id')
    movie_to_delete = Movie.query.get_or_404(movie_id)
    db.session.delete(movie_to_delete)
    db.session.commit()
    return redirect(url_for('home'))





@app.route('/add_movie', methods=["GET", "POST"])
def add_movie():
    form = AddMovies()
    if form.validate_on_submit():
        movie_title = form.movie_title.data
        response = requests.get("https://api.themoviedb.org/3/search/movie",
                                params={"api_key": API_KEY, "query": movie_title})
        data = response.json()["results"]
        return render_template("select.html", options=data)
    return render_template('add.html', form=form)


@app.route('/dataset')
def dataset():
    movie_id = request.args.get("id")
    if not movie_id:
        return "<h1>No movie ID provided!</h1>"

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if not data:
        return "<h1>Movie not found!</h1>"

    response_img = 'https://image.tmdb.org/t/p/w500/' + data['poster_path']

    new_movie = Movie(
        title=data['original_title'],
        year=data['release_date'].split('-')[0],
        description=data['overview'],
        img_url=response_img,
        user_id=current_user.id
    )

    try:
        db.session.add(new_movie)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return f"<h1>Error: {str(e)}</h1>"

    return redirect(url_for("edit", id=new_movie.id))


if __name__ == '__main__':
    app.run(debug=True, host="localhost")
