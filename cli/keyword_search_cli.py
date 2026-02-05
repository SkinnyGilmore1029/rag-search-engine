#!/usr/bin/env python3

import argparse
import json 

def load_Json_to_dict(file_path: str) -> dict:
    """Load a JSON file and return its contents as a dictionary."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
   
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            Movie_data = load_Json_to_dict("data/movies.json") # <- this is a list of dictionaries, id, title, description
            results = []
            for dicts in Movie_data["movies"]:
                if args.query.lower() in dicts["title"].lower():
                    results.append(dicts)
                    
            for i, movie in enumerate(results[:5], start=1):
                print(f"{i}. {movie['title']} (ID: {movie['id']})")
                
                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()