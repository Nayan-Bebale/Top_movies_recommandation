# 🎬 Top Movie Recommendation Web App 🍿

Welcome to the Top Movie Recommendation Web App! This Flask-based application provides personalized movie recommendations based on user input, utilizing data from the TMDb API. Users can register, log in, manage their watchlist, and receive tailored movie suggestions.

## 📑 Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## ✨ Features
- **User Authentication**: Secure registration, login, and logout functionality.
- **Movie Database Management**: Store and manage movie details such as title, year, description, rating, and user reviews.
- **Recommendation Engine**: Fetch movie data from the TMDb API and provide personalized recommendations.
- **Responsive UI**: User-friendly and responsive design using Flask-Bootstrap.

## 🎥 Prerequisites
- Basic knowledge of Python and Flask.
- TMDb API key. You can get one by creating an account on [The Movie Database (TMDb)](https://www.themoviedb.org/).
- Follow this [YouTube tutorial](https://www.youtube.com/watch?v=1xtrIEwY_zY&t=793s) to get a better understanding of the project setup.

## 🛠 Installation

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/Nayan-Bebale/Top_movies_recommandation.git
    cd Top_movies_recommandation
    ```

2. **Create a Virtual Environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables**:
    Create a `.env` file in the root directory and add your API key and secret key:
    ```
    API_KEY=your_tmdb_api_key
    SECRET_KEY=your_secret_key
    ```

5. **Set Up the Database**:
    ```sh
    flask db init
    flask db migrate
    flask db upgrade
    ```

6. **Run the Application**:
    ```sh
    flask run
    ```

## 🚀 Usage

1. **Register**: Create an account to get started.
2. **Log In**: Access your personalized movie recommendations.
3. **Add Movies**: Search and add movies to your watchlist.
4. **Edit Reviews**: Rate and review the movies in your watchlist.
5. **Get Recommendations**: Receive tailored movie recommendations based on your input.

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a Pull Request.

## 🙏 Acknowledgements

- [The Movie Database (TMDb)](https://www.themoviedb.org/) for the API.
- [Flask](https://flask.palletsprojects.com/) for the web framework.
- [Flask-Bootstrap](https://pythonhosted.org/Flask-Bootstrap/) for the UI components.
- [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy) for string matching.

## 📧 Contact

If you have any questions or feedback, feel free to reach out!

## 📺 YouTube Video

Check out the how project works and walkthrough on YouTube: [Top Movie Recommendation Web App Tutorial](https://www.youtube.com/watch?v=0lxi6SZL3u8)

