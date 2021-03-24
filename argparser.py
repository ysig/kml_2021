import argparse

def parse():
    parser = argparse.ArgumentParser(prog='', allow_abbrev=False)
    parser.add_argument('--dry', action='store_true', help='compute-accs-train-test-split')
    return parser.parse_args()