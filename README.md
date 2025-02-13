# r1-distilled-RL

create a conda env with the given req.txt file python 3.9/10 works.
the script 3200_7.py works for a H100 - 80GB machine.

This is for gsm8k dataset.

3200 is the output length tokens
7 is number of generations.

# inference
Download the LoRA checkpoint (trained for 418 steps on gsm8k) 
- each step is 7 generations and one question only.

link: [https://huggingface.co/sathvikask/r1-1.5b-RL-gsm8k](https://huggingface.co/sathvikask/r1-1.5b-RL-gsm8k) 

run inference.ipynb with your own questions.