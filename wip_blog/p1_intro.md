# 11/30

# HELLO WORLD!

So I participated in the GPU Mode hackathon in SF this November. We got absolutely creamed.
We didn’t make the top 8, and honestly there were some pretty good reasons why. Our concept wasn’t great, and we kind of tried to build two projects at once.
But that’s in the past.

You are now reading the blog of the proud owner of **$1,000 of GPU credits, baby boy.** Nebius gave everyone 1k credits to do whatever they want.

So I started scheming and landed on a ridiculous but fun idea: I want to do a **full end-to-end training run of my own LLM chatbot.** From scratch. Real training. Real GPUs. Real pain.

I have no idea how long this is going to take, but the only way forward is to start.
So let’s begin with the goals.

## WHY?

I figure every ML engineer should do this at least once. It’s like a nerdy pilgrimage.
You know how there’s that ritual where muslims go to the holy land?
This is that, but for people who think CUDA errors build character.

## Goals

By the end of this whole thing, I want to understand and implement:

1. Modern small-LLM features and layers
2. Profilers (especially PyTorch profiling)
3. A full end-to-end training stack: pretraining, midtraining, post-training
4. A somewhat optimized inference pipeline
5. A decent ablation study where I actually improve something
6. A unique personality with at least some personally sourced data

## Rules

1. I must understand every line of code. At least conceptually.
2. I have to document the journey.
3. Vibe-check the model with evals so I can actually watch loss go down and accuracy go up.
4. Everything must be open source. Weights, code, all of it.

## Who should read this?

If you've watched all of Karpathy's "0 to Hero" videos, you should be able to follow along just fine.
I’m assuming a mid-level understanding of LLMs and deep learning.

# Let's go.

Best place to start is figuring out what I need, so I made a funny little checklist:

- [ ] Model
- [ ] Data
  - [ ] Pretraining
  - [ ] Midtraining
  - [ ] Post-training
- [ ] Tokenizer
- [ ] Inference
- [ ] Checkpoint saving + loading
- [ ] Profiling
- [ ] ?
- [ ] Profit


### game plan

The order doesn't matter that much, but I like starting with the model because that’s what makes sense to me. I have a few 5090s lying around because I’m a nerd with disposable income (and I have an equally nerdy boss), so I want this to be an iterative process.

In other words, once the training pipeline is done, I'm going to tweak a bunch of model parameters based on various papers I've read and hopefully get something kind of cool at the end. Then, once I lock in my parameters, I'm gonna do a final YOLO-run.

The initial loss graph seems to drop pretty quickly, so I think I can just implement a bunch of experiments, queue them up for a day of training on the 5090s each, and see what happens.

Will I be able to beat Karpathy's performance? Probably not.
Will I have fun? Maybe.
Will I profit? No.
Will I be cracked at the end??? ...No.
Will I learn something? YES!