import sys

from textformatting import ssplit
from tqdm.auto import tqdm
import neologdn
input = sys.stdin.readline

MAX_LINES=int(sys.argv[1])

def main():
    line = 'start'

    bar = tqdm(total = MAX_LINES)
    while line:
        line = neologdn.normalize(input().rstrip())
        [print(s) for s in ssplit(line) ]
        bar.update(1)

if __name__ == "__main__":
    main()
