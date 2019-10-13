import io
import re
from contextlib import redirect_stdout
import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

for _ in range(1000):
    with open('original_pitches.txt') as out:
        f = io.StringIO()
        with redirect_stdout(f):
            gpt2.generate(sess,
                length=1000,
                top_p=0.2,
                nsamples=10,
                temperature=2.0,
            )
        content = [x.strip() for x in out.readlines()]
        for line in f.getvalue().split('\n')[1:-1]:
            line = line.strip()
            line_arr = line.split(' ')
            if all([line.lower() not in [w.lower() for w in content],
                    all(x.isalpha() or x.isspace() or x.isnumeric() or x in ['.'] for x in line),
                    len(line) < 70 and len(line) > 40,
                    line.count('.') < 2,
                    sum(list(map(lambda x:1 if x.isdigit() else 0,set(line)))) < 2,
                    all([w[-1] == w[-1].lower() for w in line.split(' ') if len(w) > 1]),
                    len(line_arr) > 3,
                    line_arr[0] != 'I',
                    line_arr[1].lower()[-1] != 's',
                    ]):
                line = line[0].upper() + line[1:]
                if line[-1] == '.':
                    line = line[:-1]
                print(line)
