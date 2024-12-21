
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
    "US":  "English",  # United States - 
    "IN":  "Hindi",  # India - 
    "FR":  "French",  # France - 
    "DE":  "German",  # Germany - 
    "JP":  "Japanese",  # Japan - 
    "CN":  "Mandarin",  # China - 
    "KR":  "Korean",  # South Korea - 
    "ES":  "Spanish",  # Spain - 
    "RU":  "Russian",  # Russia - 
}

sub_country_language_map = {
    "United States": {
        "English": "en-US",
        "Spanish": "es-US",
        "French": "fr-US",
        "Chinese": "zh-US",
        "Tagalog": "tl-US"
    },
    "India": {
        "Hindi": "hi-IN",
        "Telugu": "te-IN",
        "Tamil": "ta-IN",
        "Malayalam": "ml-IN",
        "Kannada": "kn-IN",
        "Marathi": "mr-IN",
        "Bangali": "bn-IN",
        "Gujarati": "gu-IN",
        "Panjabi": "pa-IN"
    },
    "France": {
        "French": "fr-FR",
        "Occitan": "oc-FR",
        "Breton": "br-FR",
        "Alsatian": "gsw-FR",
        "Corsican": "co-FR"
    },
    "Germany": {
        "German": "de-DE",
        "Sorbian": "wen-DE",
        "Frisian": "fy-DE",
        "Danish": "da-DE",
        "Romani": "rom-DE"
    },
    "Japan": {
        "Japanese": "ja-JP",
        "Ainu": "ain-JP",
        "Okinawan": "ryu-JP",
        "Amami": "ama-JP",
        "Miyako": "mvi-JP"
    },
    "China": {
        "Mandarin": "zh-CN",
        "Cantonese": "yue-CN",
        "Shanghainese": "wuu-CN",
        "Hokkien": "nan-CN",
        "Hakka": "hak-CN"
    },
    "South Korea": {
        "Korean": "ko-KR",
        "Jeju": "jje-KR"
    },
    "Spain": {
        "Spanish": "es-ES",
        "Catalan": "ca-ES",
        "Galician": "gl-ES",
        "Basque": "eu-ES",
        "Aranese": "oc-ES"
    },
    "Russia": {
        "Russian": "ru-RU",
        "Tatar": "tt-RU",
        "Chechen": "ce-RU",
        "Bashkir": "ba-RU",
        "Chuvash": "cv-RU"
    }
}

