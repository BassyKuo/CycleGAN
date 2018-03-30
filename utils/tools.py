from termcolor import colored

def format_time(seconds):
    s = round(seconds)
    if s < 60:         return '%2ds'                % (s)
    elif s < 60*60:    return '%2dm %02ds'          % (s / 60, s % 60)
    elif s < 24*60*60: return '%2dh %02dm %02ds'    % (s / (60*60), (s / 60) % 60, s % 60)
    else:              return '%2dd %2dh %02dm'      % (s / (24*60*60), (s / (60*60)) % 24, (s / 60) % 60)

def print_list(list, name='list'):
    print (colored("{}: ({})".format(name.upper(), len(list)), 'cyan'))
    for i in list:
        print ("  ", i)

def cuttool(total, num_split, mode='size', **kwargs):
    if total < num_split:
        num_split = total
    if mode == 'size':
        splittable = [int(total / num_split)] * num_split
        splittable[-1] += total - sum(splittable)
    elif mode == 'index':
        size_list = [int(total / num_split)] * num_split
        size_list[-1] += total - sum(size_list)
        splittable = [[sum(size_list[:i]), sum(size_list[:i+1])] for i in range(len(size_list))]
    elif mode == 'manual':
        ratio = kwargs['ratio']
        size_list = [int(total * r/sum(ratio)) for r in ratio]
        size_list[-1] += total - sum(size_list)
        splittable = [[sum(size_list[:i]), sum(size_list[:i+1])] for i in range(len(size_list))]
    else:
        raise NameError('Not support %s mode yet.' % mode)
    return splittable


# ===[ Print Colors
class bcolors:
    """
    [256-term]
        regular: \033[38;5;xxxm
        cover:   \033[48;5;xxxm
    [8-term]
        BLACK     = '\033[90m'
        RED       = '\033[91m'
        GREEN     = '\033[92m'
        YELLOW    = '\033[93m'
        BLUE      = '\033[94m'
        PINK      = '\033[95m'
        CYAN      = '\033[96m'
        WHITE     = '\033[97m'
    [special]
        END       = '\033[0m'
        BOLD      = '\033[1m'
        DARK      = '\033[2m'
        UNDERLINE = '\033[4m'
        FLASH     = '\033[5m'
    """
    END       = '\033[0m'
    BOLD      = '\033[1m'
    DARK      = '\033[2m'
    UNDERLINE = '\033[4m'
    FLASH     = '\033[5m'
    BLACK     = '\033[90m'
    RED       = '\033[91m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    BLUE      = '\033[94m'
    PINK      = '\033[95m'
    CYAN      = '\033[96m'
    WHITE     = '\033[97m'

