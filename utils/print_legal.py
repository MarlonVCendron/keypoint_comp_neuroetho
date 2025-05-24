from terminology import in_green, in_yellow, in_red

def print_legal(*args, type=None):
    if type == 'warn':
        color_func = in_yellow
    elif type == 'error':
        color_func = in_red
    else:
        color_func = in_green

    for arg in args:
        print(color_func(f'🐀 {arg}'))
