import gpt_2_simple as gpt2

model_name = "774M"
gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              'full.txt',
              model_name=model_name,
              multi_gpu=True,
              sample_length=100,
              sample_num=10,
              steps=10001)   # steps is max number of training steps

gpt2.generate(sess)
