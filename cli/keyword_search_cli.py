#!/usr/bin/env python3

import argparse
from Helpers import Commands, Search_helps




def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")


    args = parser.parse_args()
    

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = Commands.Search_Command(args.query)
            for i, movie in enumerate(results[:5], start=1):
                print(f"{i}. {movie['title']} (ID: {movie['id']})")
        case "build":
            print("Building inverted index...")

            invertedindex = Search_helps.InvertedIndex({}, {})
            invertedindex.build()
            invertedindex.save()

            # 👇 THIS IS THE ASSIGNMENT REQUIREMENT
            merida_docs = invertedindex.get_documents("merida")
            if merida_docs:
                print(f"First document ID for token 'merida': {merida_docs[0]}")
            else:
                print("Token 'merida' not found in index.")

            print("Index built and saved successfully.")
                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()