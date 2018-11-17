def add_sos_eos(inp, output):
    with open(inp) as f:
        lines = f.readlines()
    with open(output, 'w') as f:
        for line in lines:
            line = "<s> {} </s>\n".format(line.strip())
            f.write(line)
    print('done')

def main():
    input_path = '/home/alta/BLTSpeaking/ged-pm574/local/seq2seq/data/iwslt15/train.vi'
    output_path = '/home/alta/BLTSpeaking/ged-pm574/local/seq2seq/data/iwslt15/train.vi.v2'
    add_sos_eos(input_path, output_path)

if __name__ == '__main__':
    main()
