#!/usr/bin/env python3

import argparse
from Helpers import Commands, Search_helps




def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to check frequency for")
    
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to check inverse document frequency for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to check TF-IDF score for")


    args = parser.parse_args()
    

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = Commands.Search_Command(args.query)
            for i, movie in enumerate(results[:5], start=1):
                print(f"{i}. {movie['title']} (ID: {movie['id']})")
                
        case "build":
            print("Building inverted index...")

            invertedindex = Search_helps.InvertedIndex({}, {}, {})
            invertedindex.build()
            invertedindex.save()

            print("Index built and saved successfully.")
               
        case "tf":
             freq = Commands.Term_Frequency_Command(args.doc_id, args.term)
             print(f"Term Frequency of '{args.term}' in document {args.doc_id}: {freq}")
             
        case "idf":
            idf_score = Commands.Inverse_Document_Frequency_Command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_score:.2f}")
             
        case "tfidf" :
            tfidf_score = Commands.TF_IDF_Command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document {args.doc_id}: {tfidf_score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()