
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

