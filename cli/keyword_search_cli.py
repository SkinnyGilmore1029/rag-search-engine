#!/usr/bin/env python3

import argparse

from Helpers.Search_helps import InvertedIndex
from Helpers.Commands import (Search_Command, 
                              Term_Frequency_Command,   Inverse_Document_Frequency_Command, 
                              TF_IDF_Command,
                              bm25_idf_command as BM25_IDF_Command)






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

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")


    WhoMadeMe_parser = subparsers.add_parser("WhoMadeMe", help="Learn about the creator of this CLI")

    args = parser.parse_args()
    

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = Search_Command(args.query)
            for i, movie in enumerate(results[:5], start=1):
                print(f"{i}. {movie['title']} (ID: {movie['id']})")
                
        case "build":
            print("Building inverted index...")

            invertedindex = InvertedIndex({}, {}, {})
            invertedindex.build()
            invertedindex.save()

            print("Index built and saved successfully.")
               
        case "tf":
             freq = Term_Frequency_Command(args.doc_id, args.term)
             print(f"Term Frequency of '{args.term}' in document {args.doc_id}: {freq}")
             
        case "idf":
            idf_score = Inverse_Document_Frequency_Command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_score:.2f}")
             
        case "tfidf" :
            tfidf_score = TF_IDF_Command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document {args.doc_id}: {tfidf_score:.2f}")
        
        case "bm25idf":
            bm25_idf_score = BM25_IDF_Command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf_score:.2f}")
        
        case "WhoMadeMe":
            print("This CLI was created by Skinny Gilmore. From Learn Retrieval Augmented Generation on `Boot.dev` ! Make sure to check out the course if you want to learn how to build this yourself.")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()