from .Search_helps import load_movies_json, make_tokens, has_matching_token

def Search_Command(query: str) -> list[dict]:
    """Handles the 'search' command for the CLI using cached title tokens."""
    
    # Load movies
    Movie_data = load_movies_json()  # dictionary of movies
    
    # Precompute tokens for each movie title (cache)
    for movie in Movie_data:
        if "tokens" not in movie:  # only compute once
            movie["tokens"] = make_tokens(movie["title"])
    
    # Tokenize query once
    query_tokens = make_tokens(query)
    
    results: list[dict] = []
    
    # Search through cached tokens
    for movie in Movie_data:
        title_tokens = movie["tokens"]
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= 5:  # stop after first 5 matches
                break
    
    return results
