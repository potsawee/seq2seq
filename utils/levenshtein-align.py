import sys

def levenshtein_align(a,b):
    """
    align two sentences using word-level levenstein distance
    translated from Java implementation at
    https://rosettacode.org/wiki/Levenshtein_distance/Alignment#Mathematica_.2F_Wolfram_Language
        arg:
            a = ['this', 'is', 'the', 'cat']
            b = ['this', 'is', 'cat']
        output:
            a = ['this', 'is', 'the', 'cat']
            b = ['this', 'is', '***', 'cat']
    """
    costs = []
    for i in range(len(a)+1):
        x = [0] * (len(b)+1)
        costs.append(x)

    for j in range(len(b)+1):
        costs[0][j] = j
    for i in range(1, len(a)+1):
        costs[i][0] = i
        for j in range(1, len(b)+1):
            element1 = 1 + min(costs[i-1][j], costs[i][j-1])
            if a[i-1] == b[j-1]:
                element2 = costs[i-1][j-1]
            else:
                element2 = costs[i-1][j-1] + 1
            costs[i][j] = min(element1, element2)

    # walk back through matrix to figure out path
    i = len(a)
    j = len(b)
    aRev = []
    bRev = []
    while i != 0 and j != 0:
        if a[i-1] == b[j-1]:
            element = costs[i-1][j-1]
        else:
            element = costs[i-1][j-1] + 1

        if costs[i][j] == element:
            i -= 1
            j -= 1
            aRev.append(a[i])
            bRev.append(b[j])
        elif costs[i][j] == 1 + costs[i-1][j]:
            i -= 1
            aRev.append(a[i])
            bRev.append('*' * len(a[i]))
        elif costs[i][j] == 1 + costs[i][j-1]:
            j -= 1
            aRev.append('*' * len(b[j]))
            bRev.append(b[j])

    aout = aRev[::-1]
    bout = bRev[::-1]

    return aout, bout

def align_sentence(sent1, sent2):
    a = sent1.split()
    b = sent2.split()
    aout, bout = levenshtein_align(a,b)

    for a1, b1 in zip(aout, bout):
        label = 'c' if a1 == b1 else 'i'
        print('{}\t{}\t{}'.format(a1,b1,label))

def align_corpus(cor1, cor2):
    with open(cor1, 'r') as file:
        lines1 = file.readlines()
    with open(cor2, 'r') as file:
        lines2 = file.readlines()

    for sent1, sent2 in zip(lines1, lines2):
        if len(sent1.split()) > 32:
            continue
        align_sentence(sent1.strip(), sent2.strip())
        print()

def test():
    sent1 = 'the cat sat on the mat'
    sent2 = 'the cat sit on mat'
    align_sentence(sent1,sent2)

def main():
    if(len(sys.argv) != 3):
        print('Usage: python3 levenshtein-align.py original corrupted')
        return
    original = sys.argv[1]
    corrupted = sys.argv[2]
    align_corpus(original, corrupted)

if __name__ == '__main__':
    main()
