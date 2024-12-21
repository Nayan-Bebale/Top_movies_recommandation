import requests
import pickle
import pandas as pd
import logging
import os
import re
import asyncio
import aiohttp
import gzip
import json
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import UserMixin, LoginManager, login_user, logout_user, current_user, login_required, login_manager
from werkzeug.security import generate_password_hash, check_password_hash
from flask_bootstrap import Bootstrap
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from concurrent.futures import ThreadPoolExecutor
from fuzzywuzzy import process, fuzz
from tenacity import retry, stop_after_attempt, wait_fixed
from requests.exceptions import ConnectionError, RequestException
from sqlalchemy import ForeignKeyConstraint
from werkzeug.utils import secure_filename

# Import data from assemble_code
from assemble_code.parameters import GENRE, GENRE_IDS, country_language_map, stopwords, sub_country_language_map
from assemble_code.forms import SearchMovies, RegistrationForm, Edit, AddMovies 


load_dotenv()


db = SQLAlchemy()  # Create an instance of SQLAlchemy without initializing it with app.
migrate = Migrate()  # Create an instance of Migrate without initializing it with app.

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
bootstrap = Bootstrap(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movies.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


DATA_FILE = "homepage_data.json"

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


db.init_app(app)  # Initialize with the app only once
migrate.init_app(app, db)  # Initialize the migration manager with the app and db

API_KEY = os.getenv("API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
TinyURL_API_KEY = os.getenv("TinyURL_API_KEY")

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# ################################################### Database 
# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    profile_pic = db.Column(db.String(150), default='default.png')
    favorites = db.relationship('Favorite', backref='user', lazy=True)
    comments = db.relationship('Comment', backref='user', lazy=True)
    watch_history = db.relationship('WatchHistory', backref='user', lazy=True)
    critics = db.relationship('FilmCritic', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# Movie Model
class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(250), unique=True, nullable=False)
    description = db.Column(db.JSON, nullable=False)
    video_url = db.Column(db.String(250), nullable=False)
    comments = db.relationship('Comment', backref='movie', lazy=True)
    favorites = db.relationship('Favorite', backref='movie', lazy=True)
    watch_history = db.relationship('WatchHistory', backref='movie', lazy=True)
    critics = db.relationship('FilmCritic', backref='movie', lazy=True)


# Favorite Model
class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    added_on = db.Column(db.DateTime, default=datetime.utcnow)

# Comment Model
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    comment_text = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# Watch History Model
class WatchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    watched_on = db.Column(db.DateTime, default=datetime.utcnow)

# Film Critic Model
class FilmCritic(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    critic_title = db.Column(db.String(150), nullable=False)
    critic_text = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class UserPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', name='fk_userpreference_user'), nullable=False)
    preferred_genres = db.Column(db.String(500))
    preferred_language = db.Column(db.String(50), nullable=True)


class ApiCache(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer, nullable=False)
    data = db.Column(db.JSON, nullable=False)
    cached_on = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@app.template_filter('truncate_words')
def truncate_words(s, num_words):
    words = s.split()
    if len(words) > num_words:
        return ' '.join(words[:num_words]) + '...'
    return s


app.jinja_env.filters['truncate_words'] = truncate_words

############################################################# Recomendation and Fetch data 

# Get country code
def get_country_and_language():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        country_code = data.get("country", "US")
        # print(f'#################---------------------> {country_code}')
        language_code = country_language_map.get(country_code, "en-US")
        # print(f'#################---------------------> {language_code}')
        return language_code
    except Exception as e:
        print(f"Error: {e}")
        return "US", "en-US"


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
        return False

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


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def fetch_movie_details(session, movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                return await response.json()
            else:
                logging.error(f"Error {response.status}: Failed to fetch details for movie ID {movie_id}")
                return {}
    except aiohttp.ClientConnectorError as e:
        logging.error(f"Connection error while accessing {url}: {e}")
        raise
    except asyncio.TimeoutError:
        logging.error(f"Timeout error while accessing {url}")
        raise


async def fetch_country_based_movie():
    language = get_country_and_language()

    # Find the country and sub-languages
    for country, languages in sub_country_language_map.items():
        if language in languages:
            country_code = country
            language_code = languages[language]
            original_language = language_code.split('-')[0]
            
            break
    else:
        # Default if language not found
        country_code, language_code, original_language = "US", "en-US", "en"

    # Construct the TMDb API URL
    url = (f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}"
           f"&language={language_code}&region={country_code}"
           f"&with_original_language={original_language}&sort_by=popularity.desc&page=1")

    async with aiohttp.ClientSession() as session:
        try:
            # Fetch data from the API
            response = await fetch_movie_data(session, url)
            movies = response.get("results", [])
            
            return movies[:12]
        
        except ConnectionError:
            return {}
        
        except RequestException as e:
            print(f"Request error: {e}")
            return {}
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {}


async def get_homepage_data():
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Check if data file exists and is up-to-date
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as file:
            cached_data = json.load(file)
        if cached_data.get("date") == current_date:
            return cached_data["data"]

    async with aiohttp.ClientSession() as session:
        current_year = datetime.now().year

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

        urls = {**common_urls, **genre_urls}
        tasks = [fetch_movie_data(session, url) for url in urls.values()]
        results = await asyncio.gather(*tasks)
        results_dict = {key: result.get('results', []) for key, result in zip(urls.keys(), results)}

        best_movie_id = results_dict["best_movie_of_month"][0]["id"] if results_dict["best_movie_of_month"] else None
        best_movie_details = None
        if best_movie_id:
            best_movie_details = await fetch_movie_details(session, best_movie_id)
            
            youtube_url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": f"{best_movie_details['title']} trailer",
                "type": "video",
                "maxResults": 1,
                "key": YOUTUBE_API_KEY
            }
            async with session.get(youtube_url, params=params, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()
                if data.get("items"):
                    video_id = data["items"][0]["id"]["videoId"]
                    video_url = f"https://www.youtube.com/embed/{video_id}"
                


        final_data = {
            "top_3_movies": results_dict["top_3_movies"][:3],
            "latest_12_movies": results_dict["latest_12_movies"][:12],
            "upcoming_12_movies": results_dict["upcoming_12_movies"][:12],
            "director_choice": results_dict["director_choice"][:6],
            "top_10_movies": results_dict["top_10_movies"][:10],
            "best_movie_of_month": best_movie_details,
            "country_based_movies": await fetch_country_based_movie(),
            "best_movie_video_url": video_url if best_movie_details else None,
            **{key: value[:4] for key, value in results_dict.items() if key.startswith("trending_")},
        }

        with open(DATA_FILE, "w") as file:
            json.dump({"date": current_date, "data": final_data}, file, indent=4)

        return final_data


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


async def fetch_genre_based_recommendations(session, genre_ids):
    """Fetch 5 recommended movies based on genres asynchronously."""
    if not genre_ids:
        return []

    # Use TMDb Discover API to fetch movies by genre
    tmdb_discover_url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": API_KEY,
        "with_genres": ','.join(map(str, genre_ids)),
        "sort_by": "popularity.desc",
        "page": 1
    }

    try:
        async with session.get(tmdb_discover_url, params=params, timeout=10) as response:
            response.raise_for_status()
            data = await response.json()
            results = data.get('results', [])
            return [
                {
                    "id": movie.get('id'),
                    "title": movie.get('title'),
                    "year": movie.get('release_date', 'N/A').split('-')[0],
                    "description": movie.get('overview', 'No description available.'),
                    "img_url": f"https://image.tmdb.org/t/p/w500/{movie.get('poster_path')}" if movie.get('poster_path') else None,
                }
                for movie in results[:5]  # Fetch up to 5 recommendations
            ]
    except aiohttp.ClientError as e:
        print(f"Error fetching genre-based recommendations: {e}")
        return []


async def recommend_single(session, movie_details):
    movie = movie_details.get('title')

    if not movie:
        return []

    # Now we fetch recommendations asynchronously
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_details['id']}/recommendations"
        params = {"api_key": API_KEY, "language": "en-US"}
        async with session.get(url, params=params, timeout=10) as response:
            response.raise_for_status()
            data = await response.json()
            return [
                {
                    "id": recommendation.get('id'),
                    "title": recommendation.get('title'),
                    "year": recommendation.get('release_date', 'N/A').split('-')[0],
                    "description": recommendation.get('overview', 'No description available.'),
                    "img_url": f"https://image.tmdb.org/t/p/w500/{recommendation.get('poster_path')}" if recommendation.get('poster_path') else None,
                }
                for recommendation in data.get('results', [])[:5]  # Fetch up to 5 recommendations
            ]
    except aiohttp.ClientError as e:
        print(f"Error fetching movie recommendations: {e}")
        return []


############################################################# Flask Authentication
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        flash('Invalid credentials.', 'danger')
    return render_template('free_movie_zip/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Check if all fields are filled
        if not all([username, email, password, confirm_password]):
            flash('All fields are required.', 'danger')
            return render_template('free_movie_zip/register.html')

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('free_movie_zip/register.html')

        # Check if the email or username already exists
        existing_user = User.query.filter((User.email == email) | (User.username == username)).first()
        if existing_user:
            flash('User with this email or username already exists.', 'danger')
            return render_template('free_movie_zip/register.html')

        # Hash the password and create a new user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        new_user = User(username=username, email=email, password_hash=hashed_password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'danger')

    return render_template('free_movie_zip/register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


############################################################ Call this one

# Use this globally
homepage_data = asyncio.run(get_homepage_data())
best_movie_of_month =  homepage_data.get('best_movie_of_month', {})
best_movie_video_url = homepage_data.get('best_movie_video_url', None)

############################################################ Flask Routes
@app.route("/")
def home():
    languages = ['English', 'Hindi', 'Japanese', 'Korean', 'Russian', 'Mandarin (Chaina)', 'German', 'French', 'Spanish']
    form = SearchMovies()
    # Fetch homepage data asynchronously
    global homepage_data
    homepage_data = asyncio.run(get_homepage_data())
    lang = get_country_and_language()

    movie_data = []
    
    if current_user.is_authenticated:
        user_watch_history = db.session.query(Movie).join(WatchHistory).filter(WatchHistory.user_id == current_user.id).order_by(WatchHistory.watched_on.desc()).all()

        movie_data = [
            {
                'id': movie.id,
                'title': movie.title,
                'poster_path': movie.description['poster_path'],
            }
            for movie in user_watch_history
        ]

    # Prepare context for rendering the template
        # print('User watch history:', movie_data)
    context = {
        'title': 'Home',
        'top_3_movies': homepage_data.get('top_3_movies', []),
        'latest_12_movies': homepage_data.get('latest_12_movies', []),
        'upcoming_12_movies': homepage_data.get('upcoming_12_movies', []),
        'director_choice': homepage_data.get('director_choice', []),
        'top_10_movies': homepage_data.get('top_10_movies', []),
        'best_movie_of_month': homepage_data.get('best_movie_of_month', {}),
        'best_movie_video_url': homepage_data.get('best_movie_video_url', None),
        'user_watch_history': movie_data[:8],
        'country_based_movies': homepage_data.get('country_based_movies', [])[::-1],
        'GENRE_IDS':list(GENRE.values()),
        'languages': languages,
        'language': lang,
        'genre_movies': {
            genre: homepage_data.get(f"trending_{genre}", [])
            for genre in GENRE_IDS.keys()
        },
    }
    return render_template("free_movie_zip/index.html", context=context, form=form)


# Favorite System
@app.route('/add_favorite', methods=['GET', 'POST'])
@login_required
def add_favorite():
    movie_id = request.args.get('movie_id')
    movie_name = request.args.get('movie_name')
    movie = Movie.query.filter_by(id=movie_id).first()
    if not movie:
        flash("Movie not found.", "error")
        return redirect(url_for('home'))
    
    new_favorite = Favorite(user_id=current_user.id, movie_id=movie_id)
    db.session.add(new_favorite)
    db.session.commit()
    flash("Movie added to favorites!", "success")
    return redirect(url_for('movie_details', movie=movie_name, id=movie_id))

@app.route('/remove_watch_history', methods=['GET', 'POST'])
@login_required
def remove_watch_history():
    movie_id = request.args.get('movie_id')
    watch_history = WatchHistory.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
    db.session.delete(watch_history)
    db.session.commit()
    return redirect(url_for('home'))


@app.route('/my_favorites', methods=['GET'])
@login_required
def my_favorite():
    form = SearchMovies()
    
    # Query to get favorite movies with details
    favorites = db.session.query(Movie).join(Favorite).filter(Favorite.user_id == current_user.id).order_by(Favorite.added_on.desc()).all()
    
    # Prepare data for the template
    favorite_data = [
        {
            'id': movie.id,
            'title': movie.title,
            'description': movie.description,
            'video_url': movie.video_url
        }
        for movie in favorites
    ]
    
    context = {
        'best_movie_of_month': best_movie_of_month,
        'best_movie_video_url': best_movie_video_url,
        'favorites': favorite_data
    }
    return render_template('free_movie_zip/my_favorites.html', form=form, context=context)

@app.route('/remove_favorite', methods=['GET', 'POST']) 
@login_required
def remove_favorite():
    movie_id = request.args.get('movie_id')
    favorite = Favorite.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
    
    if favorite:  # Check if the favorite exists
        db.session.delete(favorite)
        db.session.commit()
    
    return redirect(url_for('my_favorite'))

# Critics System
@app.route('/my_critics', methods=['GET'])
@login_required
def my_critics():
    form = SearchMovies()

    # Query FilmCritic along with related Movie
    critics = db.session.query(FilmCritic, Movie.title).join(Movie, FilmCritic.movie_id == Movie.id).filter(FilmCritic.user_id == current_user.id).all()

    # Transform critics into a more usable format
    critic_data = [
        {
            'critic': critic,
            'movie_title': movie_title,
        }
        for critic, movie_title in critics
    ]

    context = {
        'best_movie_of_month': best_movie_of_month,
        'best_movie_video_url': best_movie_video_url,
        'critics': critic_data,
    }
    return render_template('free_movie_zip/my_critics.html', form=form, context=context)


@app.route('/add_critic', methods=['GET', 'POST'])
@login_required
def add_critic():
    form = SearchMovies()
    homepage_data = asyncio.run(get_homepage_data())
    movie_id = request.args.get('movie_id')
    movie = Movie.query.filter_by(id=movie_id).first()

    if request.method == "POST":
        newCritic = FilmCritic(
            user_id = current_user.id,
            movie_id = request.form.get('movie_id'),
            critic_title = request.form.get('critics_title'),
            rating = float(request.form.get('rating')),
            critic_text = request.form.get('critic_content')
        )
        db.session.add(newCritic)
        db.session.commit()
        flash('Critic added successfully!', 'success')
        return redirect(url_for('my_critics'))
        
    context = {
        'best_movie_of_month': homepage_data.get('best_movie_of_month', {}),
        'best_movie_video_url': homepage_data.get('best_movie_video_url', None),
        'TinyURL_API_KEY': TinyURL_API_KEY,
    }
    return render_template('free_movie_zip/add_critics.html', form=form, context=context, movie_details=movie.description)


@app.route('/delete_critic', methods=['Get', 'POST'])
@login_required
def delete_critic():
    critic_id = request.args.get('id')
    critic = FilmCritic.query.filter_by(id=critic_id).first()
    db.session.delete(critic)
    db.session.commit()
    return redirect(url_for('my_critics'))


@app.route('/edit_critic', methods=['GET', 'POST'])
@login_required
def edit_critic():
    form = SearchMovies()
    context = {
        'best_movie_of_month': best_movie_of_month,
        'best_movie_video_url': best_movie_video_url,
        'TinyURL_API_KEY': TinyURL_API_KEY,
    }


    critic_id = request.args.get('id')
    critic = FilmCritic.query.get_or_404(critic_id)

    if request.method == 'POST':
        critic_title = request.form['critics_title']
        rating = request.form['rating']
        critic_text = request.form['critic_content']
        if not critic_title or not rating or not critic_text:
            flash("All fields are required!", "error")
            return redirect(url_for('edit_critic', id=critic_id))

        try:
            critic.critic_title = critic_title
            critic.rating = int(rating)
            critic.critic_text = critic_text

            db.session.commit()
            flash("Critic updated successfully!", "success")
            return redirect(url_for('my_critics', id=critic_id))
        except Exception as e:
            db.session.rollback()
            flash(f"Error updating critic: {e}", "error")
            return redirect(url_for('edit_critic', id=critic_id))

    return render_template('free_movie_zip/edit_critic.html', critic=critic, form=form, context=context)


@app.route('/critics', methods=['GET'])
def critics():
    form = SearchMovies()
    # Fetch all critics in descending order by timestamp
    critics = db.session.query(
        FilmCritic, User.username, User.profile_pic, Movie.title.label('movie_title')
    ).join(User, FilmCritic.user_id == User.id).join(Movie, FilmCritic.movie_id == Movie.id).order_by(FilmCritic.timestamp.desc()).all()

    # Create context for rendering
    context = {
        'best_movie_of_month': best_movie_of_month,
        'best_movie_video_url': best_movie_video_url,
        'critics': [
            {
                'critic': critic,
                'username': username,
                'profile_pic': profile_pic,
                'movie_title': movie_title
            }
            for critic, username, profile_pic, movie_title in critics
        ]
    }

    return render_template('free_movie_zip/all_critics.html', context=context, form=form)

# End Critics System

@app.route('/upload-profile-pic', methods=['GET', 'POST'])
def upload_profile_pic():
    if 'profile_pic' not in request.files:
        print('Data is not store')
        return redirect(request.url)
    file = request.files['profile_pic']
    if file.filename == '' or not allowed_file(file.filename):
        print('Data is not valid or something')
        return redirect(url_for('home'))
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        current_user.profile_pic = filename
        db.session.commit()
        return redirect(url_for('home'))

@app.route("/genre", methods=["GET", "POST"])
async def genre():
    genre = request.args.get('genre', '').lower()
    genre_id = GENRE_IDS.get(genre)

    if not genre_id:
        print(f"Invalid genre: {genre}")
        return redirect('/#popular')

    # Construct the TMDb API URL
    genre_movies_url = (f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}"
                        f"&with_genres={genre_id}&sort_by=popularity.desc&page=1")

    async with aiohttp.ClientSession() as session:
        try:
            # Fetch data from the API
            response = await fetch_movie_data(session, genre_movies_url)
            # Extract movie results
            genre_movies = response.get("results", [])[:8]  # Limit to the first 8 movies
            
            # Prepare the context for the template
            context = {
                'best_movie_of_month': homepage_data.get('best_movie_of_month', {}),
                'best_movie_video_url': homepage_data.get('best_movie_video_url', None),
                'movie_genre': genre.capitalize(),
                'genre_movies': genre_movies,
            }
            return render_template("free_movie_zip/genre.html", context=context, form=SearchMovies())

        except aiohttp.ClientConnectionError:
            print("Network error: Unable to connect to TMDb API.")
            return redirect('/#popular')

        except aiohttp.ClientResponseError as e:
            print(f"HTTP error during TMDb API call: {e}")
            return redirect('/#popular')

        except asyncio.TimeoutError:
            print("The request to TMDb API timed out.")
            return redirect('/#popular')

        except Exception as e:
            # Catch-all for unexpected errors
            print(f"Unexpected error: {e}")
            return redirect('/#popular')


@app.route("/language", methods=["GET", "POST"])
async def go_languages():
    language = request.args.get('language')
    sub_languages = []

    # Find the country and sub-languages
    for country, languages in sub_country_language_map.items():
        if language in languages:
            country_code = country
            language_code = languages[language]
            original_language = language_code.split('-')[0]
            
            # Extract sub-languages excluding the selected one
            sub_languages = [lang for lang in languages if lang != language]
            
            break
    else:
        # Default if language not found
        country_code, language_code, original_language = "US", "en-US", "en"

    # Construct the TMDb API URL
    url = (f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}"
           f"&language={language_code}&region={country_code}"
           f"&with_original_language={original_language}&sort_by=popularity.desc&page=1")

    async with aiohttp.ClientSession() as session:
        try:
            # Fetch data from the API
            response = await fetch_movie_data(session, url)
            movies = response.get("results", [])
            
            context = {
                'best_movie_of_month': homepage_data.get('best_movie_of_month', {}),
                'best_movie_video_url': homepage_data.get('best_movie_video_url', None),
                'language': language,
                'sub_languages': sub_languages,  # Include sub-languages
                'movies': movies[:8],  # Limit to the first 8 movies
            }
            return render_template("free_movie_zip/language.html", context=context, form=SearchMovies())
        
        except ConnectionError:
            return redirect('/#stream')
        
        except RequestException as e:
            print(f"Request error: {e}")
            return redirect('/#stream')
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            return redirect('/#stream')


@app.route("/movie-details", methods=["GET", "POST"])
async def movie_details():
    movie_id = request.args.get('id')
    video_url = None
    movie_details = {}
    recommended_movies = []

    if current_user.is_authenticated and movie_id:
        if WatchHistory.query.filter_by(user_id=current_user.id, movie_id=movie_id).first() is None:
            new_watch = WatchHistory(user_id=current_user.id, movie_id=movie_id)
            db.session.add(new_watch)
            db.session.commit()
    
    context = {
        'best_movie_of_month': best_movie_of_month,
        'best_movie_video_url': best_movie_video_url,
    }
    movie = Movie.query.filter_by(id=movie_id).first()
    if not movie and movie_id:
        async with aiohttp.ClientSession() as session:
            try:
                tmdb_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
                params = {"api_key": API_KEY, "language": "en-US"}
                async with session.get(tmdb_url, params=params, timeout=10) as response:
                    response.raise_for_status()
                    movie_details = await response.json()
                    new_movie = Movie(
                        id=movie_details['id'],
                        title=movie_details['title'],
                        description=movie_details,
                        video_url=""
                    )
                    db.session.add(new_movie)
                    db.session.commit()
                    movie = new_movie

                youtube_url = "https://www.googleapis.com/youtube/v3/search"
                params = {
                    "part": "snippet",
                    "q": f"{movie.title} trailer",
                    "type": "video",
                    "maxResults": 1,
                    "key": YOUTUBE_API_KEY
                }
                async with session.get(youtube_url, params=params, timeout=10) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if data.get("items"):
                        video_id = data["items"][0]["id"]["videoId"]
                        video_url = f"https://www.youtube.com/embed/{video_id}"

                        # Save video_url to the database
                        movie.video_url = video_url
                        db.session.commit()

                recommended_movies = await recommend_single(session, movie_details)
                if not recommended_movies:
                    genre_ids = [genre['id'] for genre in movie_details.get('genres', [])]
                    recommended_movies = await fetch_genre_based_recommendations(session, genre_ids)

                comments = Comment.query.filter_by(movie_id=movie.id).all()
                return render_template(
                    "free_movie_zip/movie_details.html",
                    form=SearchMovies(),
                    video_url=video_url,
                    movie_details=movie_details,
                    recommended_movies=recommended_movies,
                    comments=comments,
                    context=context
                
                )
            except Exception as e:
                print(f"Unexpected error: {e}")
                return redirect('/')

    comments = Comment.query.filter_by(movie_id=movie.id).all()
    movie_details=movie.description
    async with aiohttp.ClientSession() as session:
        recommended_movies = await recommend_single(session, movie.description)
        if not recommended_movies:
            genre_ids = [genre['id'] for genre in movie.description.get('genres', [])]
            recommended_movies = await fetch_genre_based_recommendations(session, genre_ids)
                
    return render_template(
        "free_movie_zip/movie_details.html",
        form=SearchMovies(),
        video_url=movie.video_url,
        movie_details=movie.description,
        recommended_movies=recommended_movies,
        comments=comments,
        context=context
    )


@app.route("/search", methods=["GET", "POST"])
async def search():
    form = SearchMovies()
    
    context = {
        'best_movie_of_month': best_movie_of_month,
        'best_movie_video_url': best_movie_video_url,
    }

    if form.validate_on_submit():
        movie_title = form.movie_title.data
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": API_KEY, "query": movie_title}

        async with aiohttp.ClientSession() as session:
            try:
                # Make asynchronous API call
                async with session.get(search_url, params=params, timeout=10) as response:
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    data = await response.json()
                    movies = data.get("results", [])[:5]  # Limit to the first 5 results

                # Render the search results page
                return render_template(
                    "free_movie_zip/search.html",
                    options=movies,
                    GENRE_IDS=GENRE,
                    form=form,
                    context=context
                )
            except aiohttp.ClientConnectionError:
                print("Network error: Unable to connect to TMDb API.")
                return redirect(url_for('home'))

            except aiohttp.ClientResponseError as e:
                print(f"HTTP error during TMDb API call: {e}")
                return redirect(url_for('home'))

            except asyncio.TimeoutError:
                print("The request to TMDb API timed out.")
                return redirect(url_for('home'))

            except Exception as e:
                print(f"Unexpected error: {e}")
                return redirect(url_for('home'))

    # Redirect to the home page if form is not valid
    return redirect(url_for('home'))


@app.route("/add_comment", methods=["POST"])
def add_comment():
    comment_text = request.form.get('comment_text')
    movie_id = request.form.get('movie_id')  # Use POST for security
    movie_name = request.form.get('movie_name')


    if not current_user.is_authenticated:
        flash('Please log in to leave a comment.', 'danger')
        return redirect(url_for('login'))
    

    # Create a new comment object
    new_comment = Comment(
        user_id=current_user.id,
        movie_id=movie_id,
        comment_text=comment_text.strip()
    )

    try:
        db.session.add(new_comment)
        db.session.commit()
        flash('Comment added successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding comment: {str(e)}', 'danger')

    return redirect(url_for('movie_details', movie=movie_name, id=movie_id))


@app.route("/about")
def about():
    context = {
        'best_movie_of_month': best_movie_of_month,
        'best_movie_video_url': best_movie_video_url,
    }
    return render_template("free_movie_zip/about.html", form = SearchMovies(), context=context)

@app.route("/contact")
def contect():
    context = {
        'best_movie_of_month': best_movie_of_month,
        'best_movie_video_url': best_movie_video_url,
    }
    return render_template("free_movie_zip/contact.html", form = SearchMovies(), context=context)

@app.route("/blog")
def blog():
    context = {
        'best_movie_of_month': best_movie_of_month,
        'best_movie_video_url': best_movie_video_url,
    }
    return render_template("free_movie_zip/blog.html", form = SearchMovies(), context=context)

@app.route("/blog-details")
def blog_details():
    context = {
        'best_movie_of_month': best_movie_of_month,
        'best_movie_video_url': best_movie_video_url,
    }
    return render_template("free_movie_zip/blog_detail.html", form = SearchMovies(), context=context)

@app.route("/services")
def services():
    context = {
        'best_movie_of_month': best_movie_of_month,
        'best_movie_video_url': best_movie_video_url,
    }
    return render_template("free_movie_zip/services.html", form = SearchMovies(), context=context)

@app.route("/teams")
def teams():
    context = {
        'best_movie_of_month': best_movie_of_month,
        'best_movie_video_url': best_movie_video_url,
    }
    return render_template("free_movie_zip/team.html", form = SearchMovies(), context=context)


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


@app.route('/try')
async def try_it():
    language = get_country_and_language()
    sub_languages = []

    # Find the country and sub-languages
    for country, languages in sub_country_language_map.items():
        if language in languages:
            country_code = country
            language_code = languages[language]
            original_language = language_code.split('-')[0]
            
            # Extract sub-languages excluding the selected one
            sub_languages = [lang for lang in languages if lang != language]
            
            break
    else:
        # Default if language not found
        country_code, language_code, original_language = "US", "en-US", "en"

    # Construct the TMDb API URL
    url = (f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}"
           f"&language={language_code}&region={country_code}"
           f"&with_original_language={original_language}&sort_by=popularity.desc&page=1")

    async with aiohttp.ClientSession() as session:
        try:
            # Fetch data from the API
            response = await fetch_movie_data(session, url)
            movies = response.get("results", [])
            # print(movies[:8])
            
            context = {
                'best_movie_of_month': homepage_data.get('best_movie_of_month', {}),
                'best_movie_video_url': homepage_data.get('best_movie_video_url', None),
                'language': language,
                'sub_languages': sub_languages,  # Include sub-languages
                'movies': movies[:8],  # Limit to the first 8 movies
            }
            return render_template("free_movie_zip/try.html", context=context, form=SearchMovies())
        
        except ConnectionError:
            return redirect('/#stream')
        
        except RequestException as e:
            print(f"Request error: {e}")
            return redirect('/#stream')
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            return redirect('/#stream')



if __name__ == '__main__':
    app.run(debug=True, host="localhost")