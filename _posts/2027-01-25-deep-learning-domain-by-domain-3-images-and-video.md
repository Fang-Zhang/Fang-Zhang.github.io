---
title: "Deep Learning, Domain by Domain, Part 3: Images and Video"
date: 2027-01-25 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [deep-learning, computer-vision, image-segmentation, diffusion-models, video-generation, series]
description: "Part 3 of Deep Learning, Domain by Domain. Seven image and video tasks, from classification to video generation, each tied to the paper that defined or advanced it, with the figures those papers report."
---

This post does one thing: it walks through the main tasks deep learning is applied to in images and video, in the order the work appeared, and ties each to the paper that defined or advanced it. The model families behind the tasks were introduced in [Part 2](/posts/deep-learning-domain-by-domain-2-foundations-from-convolution-and-recurrence-to-the-transformer/); this part is about what they were asked to do.

## The field: one input, many outputs

An image is a grid of numbers, and a video is a sequence of such grids. What differs between tasks is the output: a label, a label per pixel, a sentence, an answer, a score, or a new image. The figure lists the tasks covered here, and the paper behind each.

![A table of seven image and video tasks (classification, segmentation, captioning, visual question answering, image-text matching, image generation, video generation), each with its input and output and the work and reported figure covered in this post.]({{ "/assets/pic/dl-domain-3-image-tasks.png" | relative_url }})
*The seven tasks of this post. The reported figures are each source's own, measured on its own benchmark, so they are not comparable across rows. The table is the author's.*

## Classification: one label per image

Classification was covered in Parts 1 and 2: the 2012 ImageNet entry with a top-5 error of 15.3%, and the 2015 residual networks with 3.57% for an ensemble on the ImageNet test set. Every task below starts from the features such networks learn.

## Segmentation: a label for every pixel

Long, Shelhamer and Darrell's [fully convolutional networks](https://arxiv.org/abs/1411.4038) (November 2014) take input of arbitrary size and produce a correspondingly sized output, trained end to end, pixels to pixels. They adapt AlexNet, VGG and GoogLeNet into fully convolutional networks and fine-tune them for segmentation. The abstract reports a 62.2% mean IU on PASCAL VOC 2012, a 20% relative improvement, with inference taking one third of a second for a typical image.

He, Gkioxari, Dollar and Girshick's [Mask R-CNN](https://arxiv.org/abs/1703.06870) (March 2017) extends Faster R-CNN with a branch that predicts a mask for each detected object, alongside the existing bounding-box branch. Its abstract says it runs at 5 frames per second and reports top results in all three tracks of the COCO challenges: instance segmentation, bounding-box object detection and person keypoint detection.

Kirillov and colleagues' [Segment Anything](https://arxiv.org/abs/2304.02643) (April 2023) changes what the model is asked for. It is built to be promptable, so it can transfer zero-shot to new image distributions and tasks, and its data collection loop produced a dataset of over 1 billion masks on 11 million licensed and privacy-respecting images.

## Captioning and question answering: images to language

Vinyals, Toshev, Bengio and Erhan's [Show and Tell](https://arxiv.org/abs/1411.4555) (November 2014) combines a vision network with a recurrent language model, trained to maximise the likelihood of the target sentence given the image. On the Pascal dataset the abstract reports a BLEU-1 score of 59, against 25 for the previous state of the art and about 69 for human performance, and a BLEU-4 of 27.7 on COCO.

Agrawal and colleagues' [visual question answering](https://arxiv.org/abs/1505.00468) (2015) proposes a task where the input is an image and a free-form question, and the output is an answer. Their abstract argues that a system that succeeds typically needs more detailed image understanding and complex reasoning than one producing generic captions, and that answers are short enough to evaluate automatically. They provide a dataset of about 0.25 million images, 0.76 million questions and 10 million answers.

## Image-text matching: tasks specified in words

Radford and colleagues' [CLIP](https://arxiv.org/abs/2103.00020) (February 2021) starts from a limit of the classification setup: systems trained to predict a fixed set of categories need additional labelled data to handle any other visual concept. CLIP is pre-trained to predict which caption goes with which image, on 400 million image-text pairs from the internet. Natural language is then used to refer to visual concepts, which allows zero-shot transfer: the abstract reports that it matches the accuracy of the original ResNet-50 on ImageNet without using any of its 1.28 million training examples, and benchmarks it on more than 30 datasets.

## Image generation: from adversarial games to denoising

Goodfellow and colleagues' [generative adversarial nets](https://arxiv.org/abs/1406.2661) (June 2014) train two models at once: a generator that captures the data distribution, and a discriminator that estimates whether a sample came from the training data or from the generator. The generator is trained to maximise the probability that the discriminator makes a mistake, a minimax game; with multilayer perceptrons the whole system can be trained with backpropagation.

Ho, Jain and Abbeel's [denoising diffusion probabilistic models](https://arxiv.org/abs/2006.11239) (June 2020) use a different class of model, diffusion probabilistic models, and report an Inception score of 9.46 and a state-of-the-art FID score of 3.17 on unconditional CIFAR10, and sample quality similar to ProgressiveGAN on 256x256 LSUN.

Rombach and colleagues' [latent diffusion models](https://arxiv.org/abs/2112.10752) (December 2021) address the cost. Their abstract says diffusion models that operate directly in pixel space often consume hundreds of GPU days to optimise and are expensive at inference. They apply the models in the latent space of a pretrained autoencoder instead, add cross-attention layers so that the generation can be conditioned on inputs such as text or bounding boxes, and report a new state of the art for image inpainting and highly competitive results on other tasks, at significantly reduced computational cost.

## Video generation: a sequence of images, generated together

Ho and colleagues' [video diffusion models](https://arxiv.org/abs/2204.03458) (April 2022) extend the standard image diffusion architecture to video, train jointly on image and video data, and report what the authors call the first results on a large text-conditioned video generation task. OpenAI's February 2024 report [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) describes Sora, a text-conditional diffusion model trained jointly on videos and images of variable durations, resolutions and aspect ratios, using a transformer that operates on spacetime patches of video and image latent codes. Its largest model is described as capable of generating a minute of high fidelity video. The report says that model and implementation details are not included, and that Sora has numerous limitations as a simulator: it does not accurately model the physics of many basic interactions, such as glass shattering, and other interactions, such as eating food, do not always produce correct changes in object state.

## What the sources show

These points are the author's reading of the sources above, not claims the sources make.

- **The output grew from a label to language and then to the image itself.** The sequence runs from a category, to a pixel mask, to a sentence, to a prompt that specifies the task, to a generated image or video.
- **Each generation step is stated in the sources as a response to a cost or a limit.** The CLIP abstract names the fixed-category limit, the latent diffusion abstract names the GPU-day cost, and the Sora report names the failures of its own physics.
- **Part 2's designs recur.** Convolutional networks carry segmentation in 2014, a recurrent language model carries captioning in 2014, and attention on patches carries the 2024 video model.

## Limits of this summary

- It covers seven tasks. Object detection beyond the Mask R-CNN mention, image restoration, three-dimensional vision and the application areas (medicine in Part 8, vehicles in Part 5) are left out.
- Every paper except the Sora report was read as an abstract page only; the Sora report page was read in the sections on method and limitations, not as a full technical document. Figures quoted are those the sources state, on their own benchmarks.
- Benchmarks change and models are replaced quickly in this field; this post makes no claim about which model is the best today.
