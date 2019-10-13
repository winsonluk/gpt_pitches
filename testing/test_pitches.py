import io
import re
from contextlib import redirect_stdout
import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

#gpt2.generate(sess,
#        length=1000,
##        nsamples=20,
##        batch_size=5,
#        top_p=0.4,
#        temperature=1.6,
##        prefix="We",
##        truncate="\n",
#        )

START_P = 0.7
START_T = 0.5

for i in [x * 0.1 for x in range(0, 6) if (START_P + x * 0.1) <= 1]:
    for j in [x * 0.2 for x in range(0, 7)]:
        print('top_p: ' + str(START_P + i) + '\ntemp: ' + str(START_T + j))
        f = io.StringIO()
        with redirect_stdout(f):
            gpt2.generate(sess,
                length=1000,
                top_p=START_P + i,
                temperature=START_T + j,
            )
        with open('full.txt') as out:
            content = [x.strip() for x in out.readlines()]
            counter = 0
            for line in f.getvalue().split('\n')[1:-1]:
                line = line.strip()
                if all([line.lower() not in [w.lower() for w in content],
                        all(x.isalpha() or x.isspace() or x.isnumeric() or x in ['.'] for x in line),
                        len(line) < 70 and len(line) > 30,
                        line.split(' ')[0] != 'I',
                        line.split(' ')[1].lower() != 'is',
                        line.split(' ')[1].lower() != 'helps',
                        ]):
                    print(line)
                    counter += 1
            print('Results: ' + str(counter))
