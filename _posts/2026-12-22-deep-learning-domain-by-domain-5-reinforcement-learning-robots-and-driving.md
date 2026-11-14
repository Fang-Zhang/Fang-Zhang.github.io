---
title: "Deep Learning, Domain by Domain, Part 5: Reinforcement Learning, Robots and Driving"
date: 2026-12-22 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [deep-learning, reinforcement-learning, robotics, self-driving, series]
description: "Part 5 of Deep Learning, Domain by Domain. Two reinforcement learning methods, five robot-learning papers and two driving papers, with what each reports and what each says it costs."
---

This post does one thing: it follows deep learning where the output is an action, not a label: reinforcement learning from pixels, robot manipulation, and driving. It covers nine papers, states what each reports, and records what each says about the cost of its data. The image models behind the perception step were covered in [Part 3](/posts/deep-learning-domain-by-domain-3-images-and-video/), and the Transformer used in the last robotics papers in [Part 2](/posts/deep-learning-domain-by-domain-2-foundations-from-convolution-and-recurrence-to-the-transformer/).

## The field: from perceiving to acting

A classifier is scored on one image at a time. A controller acts, the world changes, and the next input depends on the action. Two methods dominate the learning side: reinforcement learning, where an agent learns from rewards, and imitation, where it learns from demonstrations. The figure shows the choice the robotics and driving papers address: a stack of separate hand-built stages, or one network from pixels to action.

![Two diagrams. A: a chain of six stages from camera through perception, state estimation, planning and low-level control to motors. B: camera pixels into one neural network into an output, with four papers listed by what the output is: motor torques, a steering command, a grasp-success probability, and actions written as text tokens.]({{ "/assets/pic/dl-domain-5-end-to-end.png" | relative_url }})
*A modular control stack against an end-to-end learned policy, with the output each paper's network produces. Stage names and layout are the author's.*

## Reinforcement learning from raw pixels

Mnih and colleagues' [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (December 2013) presents what its abstract calls the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. It is a convolutional network trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. Applied to seven Atari 2600 games with no change of architecture or algorithm, it outperforms all previous approaches on six and surpasses a human expert on three. Games are the subject of Part 7.

Schulman and colleagues' [proximal policy optimization](https://arxiv.org/abs/1707.06347) (July 2017) is a family of policy gradient methods that alternate between sampling data through interaction with the environment and optimising a surrogate objective. Where standard policy gradient methods take one update per data sample, the new objective allows multiple epochs of minibatch updates. The abstract says it keeps some benefits of trust region policy optimization while being much simpler to implement, more general, and having better sample complexity empirically.

## Robot manipulation

Levine, Finn, Darrell and Abbeel's [End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702) (April 2015) poses the question directly: does training perception and control jointly end to end perform better than training each component separately? Their policies map raw image observations directly to torques at the robot's motors, and are convolutional networks with 92,000 parameters. They are trained by guided policy search, which turns policy search into supervised learning with supervision from a simple trajectory-centric reinforcement learning method. They evaluate on real-world manipulation tasks such as screwing a cap onto a bottle.

Levine, Pastor, Krizhevsky and colleagues' [hand-eye coordination for grasping](https://arxiv.org/abs/1603.02199) (March 2016) trains a large convolutional network to predict the probability that a gripper motion will result in a successful grasp, from monocular camera images and independently of camera calibration or the current robot pose. The network then servos the gripper in real time. To train it they collected over 800,000 grasp attempts over two months, using between 6 and 14 robotic manipulators at any given time.

OpenAI's [Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) (October 2019) takes the opposite route to data. Its abstract says the models are trained only in simulation and used on a real robot, through automatic domain randomization, which generates a distribution over randomised environments of ever-increasing difficulty. The result is a humanoid robot hand that solves a Rubik's cube, involving both control and state estimation.

Brohan and colleagues' [RT-1](https://arxiv.org/abs/2212.06817) (December 2022) is a Transformer-based model for real-world robot control. The paper reports training on over 130,000 demonstration episodes collected over 17 months with 13 robots. It reports performing over 700 training instructions at a 97% success rate, and generalising to new tasks, distractors and backgrounds 25%, 36% and 18% better than the next best baseline.

Brohan and colleagues' [RT-2](https://arxiv.org/abs/2307.15818) (July 2023) fine-tunes vision-language models on both robot trajectory data and Internet-scale vision-language tasks. To fit actions and language into one format, the actions are expressed as text tokens and added to the training set the way words are. The authors call such models vision-language-action models, and report from 6,000 evaluation trials improved generalisation to novel objects, the ability to interpret commands absent from the robot training data, and rudimentary reasoning, such as picking up the smallest or largest object.

## Driving

Bojarski and colleagues' [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) (April 2016) trains a convolutional network to map raw pixels from a single front-facing camera directly to steering commands, with the human steering angle as the only training signal. The abstract reports that, with minimum training data from humans, the system learns to drive in traffic on local roads with or without lane markings, on highways, and in parking lots and on unpaved roads. It runs at 30 frames per second. The authors argue that optimising all steps simultaneously will eventually give better performance and smaller systems than explicit decomposition into lane detection, path planning and control. They state this as an argument; the abstract does not report a measured comparison.

Sun and colleagues' [Waymo Open Dataset](https://arxiv.org/abs/1912.04838) (December 2019) addresses the data side. Its abstract says existing self-driving datasets are limited in the scale and variation of the environments they capture, even though generalisation within and between operating regions is crucial. The dataset has 1,150 scenes of 20 seconds each, with synchronised LiDAR and camera data across urban and suburban geographies, and by the authors' proposed diversity metric is 15 times more diverse than the largest camera-plus-LiDAR dataset then available.

## What the sources show

These points are the author's reading of the sources above, not claims the sources make.

- **The recurring question is the one in the figure.** Levine's 2015 paper and Bojarski's 2016 paper both set a jointly trained network against hand-built stages, and RT-2 extends the same idea by writing actions as text.
- **Data collection is the stated cost.** The grasping paper reports two months and up to 14 robots, RT-1 reports 17 months and 13 robots, the Rubik's cube paper avoids real-world data by training in simulation, and the Waymo abstract says representative real-world data is resource-intensive to obtain.
- **Reported results are on specific tasks.** An Atari score, a 97% success rate on 700 instructions and a drive on local roads are different measures, and none of the abstracts or summaries above supports a claim about general reliability.

## Limits of this summary

- It covers nine papers. Simulation platforms, legged and aerial robots, motion planning, safety standards and regulation of vehicles are left out.
- Every paper was read as an abstract page, and the RT-1 paper additionally in the introduction and figure captions. The figures quoted are those the sources state, on their own tasks.
- No claim is made here about how any deployed vehicle or robot works.
