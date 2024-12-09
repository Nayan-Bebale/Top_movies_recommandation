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
from requests.exceptions import ConnectionError, RequestException


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
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Define common stopwords
stopwords = {"the", "a", "of", "and", "to", "in", "for", "with", "on", "at", "by", "an", "-", "as"}

# Genre mapping for dynamic URL construction
GENRE_IDS = {
    "action": 28,
    "adventure": 12,
    "animation": 16,
    "comedy": 35,
    "crime": 80,
    "documentary": 99,
    "drama": 18,
    "family": 10751,
    "fantasy": 14,
    "history": 36,
    "horror": 27,
    "music": 10402,
    "mystery": 9648,
    "romance": 10749,
    "science_fiction": 878,
    "thriller": 53,
    "war": 10752,
    "western": 37
}

GENRE = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    53: "Thriller",
    10752: "War",
    37: "Western"
}

country_language_map = {
    "English": ("US", "en-US"),  # United States - English
    "Hindi": ("IN", "hi-IN"),  # India - Hindi
    "French": ("FR", "fr-FR"),  # France - French
    "German": ("DE", "de-DE"),  # Germany - German
    "Japanese": ("JP", "ja-JP"),  # Japan - Japanese
    "Mandarin": ("CN", "zh-CN"),  # China - Mandarin
    "Korean": ("KR", "ko-KR"),  # South Korea - Korean
    "Spanish": ("ES", "es-ES"),  # Spain - Spanish
    "Russian": ("RU", "ru-RU"),  # Russia - Russian
}


# ################################################### Database 

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

################################################## Flask Form

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
        return country_code, language_code
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



async def get_homepage_data():
    from datetime import datetime
    current_year = datetime.now().year
    # current_year_month = datetime.now().strftime("%Y-%m")
    # start_date = f"{current_year_month}-01"
    # end_date = datetime.now().strftime("%Y-%m-%d")
    # country_code, language_code  = get_country_and_language()
    # print(country_code, language_code)

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

        # Combine URLs
        urls = {**common_urls, **genre_urls}

        # Fetch data asynchronously
        tasks = [fetch_movie_data(session, url) for url in urls.values()]
        results = await asyncio.gather(*tasks)

        # Map results back to their keys
        results_dict = {key: result.get('results', []) for key, result in zip(urls.keys(), results)}

        # Get detailed info for "best_movie_of_month"
        best_movie_id = results_dict["best_movie_of_month"][0]["id"] if results_dict["best_movie_of_month"] else None
        best_movie_details = None
        if best_movie_id:
            best_movie_details = await fetch_movie_details(session, best_movie_id)

        # Limit and return results
        return {
            "top_3_movies": results_dict["top_3_movies"][:3],
            "latest_12_movies": results_dict["latest_12_movies"][:12],
            "upcoming_12_movies": results_dict["upcoming_12_movies"][:12],
            "director_choice": results_dict["director_choice"][:6],
            "top_10_movies": results_dict["top_10_movies"][:10],
            "best_movie_of_month": best_movie_details,
            **{key: value[:4] for key, value in results_dict.items() if key.startswith("trending_")},
        }


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
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('free_movie_zip/register.html', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


############################################################ Flask Routes

@app.route("/")
def home():
    languages = ['English', 'Hindi', 'Japanese', 'Korean', 'Russian', 'Mandarin (Chaina)', 'German', 'French', 'Spanish']
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
        'GENRE_IDS':list(GENRE.values()),
        'languages': languages,
        'genre_movies': {
            genre: homepage_data.get(f"trending_{genre}", [])
            for genre in GENRE_IDS.keys()
        },
    }
    return render_template("free_movie_zip/index.html", context=context, form=form)


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
    country_code, language_code = country_language_map.get(language, ("US", "en-US"))
    original_language = language_code.split('-')[0]  # Extract the language part (e.g., "en" from "en-US")
    
    # Construct the TMDb API URL
    url = (f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}"
           f"&language={language_code}&region={country_code}"
           f"&with_original_language={original_language}&sort_by=popularity.desc&page=1")
    
    async with aiohttp.ClientSession() as session:
        try:
            # Fetch data from the API
            response = await fetch_movie_data(session, url)
            # Extract movie results
            movies = response.get("results", [])
            context = {
                'language': language,
                'movies': movies[:8],  # Limit to the first 8 movies
            }
            return render_template("free_movie_zip/language.html", context=context, form=SearchMovies())
        except ConnectionError:
            return redirect('/#stream')
        except RequestException as e:
            # Handle any other requests-related exceptions
            print(f"Request error: {e}")
            return redirect('/#stream')
        except Exception as e:
            # Catch-all for any other exceptions
            print(f"Unexpected error: {e}")
            return redirect('/#stream')


@app.route("/movie-details", methods=["GET", "POST"])
async def movie_details():
    movie_name = request.args.get('movie')
    movie_id = request.args.get('id')  # Get movie ID from request
    video_url = None
    movie_details = {}
    recommended_movies = []

    async with aiohttp.ClientSession() as session:
        try:
            # Fetch trailer from YouTube API
            if movie_name:
                youtube_url = "https://www.googleapis.com/youtube/v3/search"
                params = {
                    "part": "snippet",
                    "q": f"{movie_name} trailer",
                    "type": "video",
                    "maxResults": 1,  # Fetch only the top result
                    "key": YOUTUBE_API_KEY
                }
                async with session.get(youtube_url, params=params, timeout=10) as response:
                    response.raise_for_status()  # Raise HTTP errors if any
                    data = await response.json()
                    if data.get("items"):
                        # Extract video ID
                        video_id = data["items"][0]["id"]["videoId"]
                        video_url = f"https://www.youtube.com/embed/{video_id}"

            # Fetch movie details from TMDb API
            if movie_id:
                tmdb_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
                params = {"api_key": API_KEY, "language": "en-US"}
                async with session.get(tmdb_url, params=params, timeout=10) as response:
                    response.raise_for_status()
                    movie_details = await response.json()


                # Fetch recommended movies
                recommended_movies = await recommend_single(session, movie_details)

                # If no recommendations, fetch 5 movies from a similar genre
                if not recommended_movies:
                    genre_ids = [genre['id'] for genre in movie_details.get('genres', [])]
                    recommended_movies = await fetch_genre_based_recommendations(session, genre_ids)

            # Render the movie details page
            return render_template(
                "free_movie_zip/movie_details.html",
                form=SearchMovies(),
                video_url=video_url,  # Pass video trailer URL
                movie_details=movie_details,  # Pass detailed movie data
                recommended_movies=recommended_movies  # Pass recommendations
            )

        except aiohttp.ClientConnectionError:
            print("Network error: Unable to connect to the API.")
            return redirect('/')

        except aiohttp.ClientResponseError as e:
            print(f"HTTP error: {e}")
            return redirect('/')

        except Exception as e:
            print(f"Unexpected error: {e}")
            return redirect('/')


@app.route("/search", methods=["GET", "POST"])
async def search():
    form = SearchMovies()

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
                    form=form
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
def try_it():
    return render_template("free_movie_zip/try.html", form = SearchMovies())
if __name__ == '__main__':
    app.run(debug=True, host="localhost")