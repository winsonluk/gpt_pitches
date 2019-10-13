import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

gpt2.generate(sess,
        length=1000,
#        nsamples=20,
#        batch_size=5,
        top_p=0.4,
        temperature=1.6,
#        prefix="We",
#        truncate="\n",
        )

