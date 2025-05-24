from terminology import in_green

def print_legal(*args):
    for arg in args:
        print(in_green(f'🐀 {arg}'))
