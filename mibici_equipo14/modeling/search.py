"""Puntos de entrada de búsqueda y reproducción independiente."""
import argparse
from .advanced import search, reproduce

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--question',choices=['all','p1','p2','p3'],default='all')
    parser.add_argument('--reproduce',action='store_true')
    args=parser.parse_args()
    (reproduce if args.reproduce else search)(args.question)
