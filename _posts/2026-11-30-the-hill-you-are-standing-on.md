---
title: "How to Think Clearly (10/11): The Hill You Are Standing On"
date: 2026-11-30 06:00:00 +1300
categories: [Personal, Philosophy]
tags: [optimization, local-maximum, decision-making, thinking-series]
description: "Taking the best step available each time ends on the nearest peak, not the highest one. What that costs, how to notice it, and how to budget the descent."
---

**Where this sits:** stage 5 of 6, Direction. The [previous post](/posts/explore-early-exploit-late-imagine-the-failure-first/) said to explore early and exploit late. This one is about the situation where exploiting has worked too well: a path that keeps improving, up to a point from which every nearby step is worse.

![The six stages of the series with stage 5, Direction, highlighted](/assets/pic/thinking-series-post-10.png)
_Where this post sits in the series: stage 5 of 6, Direction._

The default way to improve anything is to look at the options in reach and take the best one. Repeated, this is hill climbing, and it works until it reaches a summit. At a summit, every available step goes down, so the method halts. Nothing in the method says whether the summit is the highest one around.

## The shape of the problem

![A one-dimensional landscape with several peaks of different heights; a lower peak is marked as a local maximum, a higher peak as the global maximum, and the valley between them as the descent](/assets/pic/local-vs-global-maximum.png)
_A sketch: from the marked local maximum, every nearby step is downhill, and the higher peak lies across a valley. Own illustration._

Optimization has names for this. In their 1983 paper [Optimization by Simulated Annealing](https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/TemperAnneal/KirkpatrickAnnealScience1983.pdf), Scott Kirkpatrick, C. D. Gelatt and M. P. Vecchi describe the usual method, iterative improvement, as accepting only rearrangements that lower the cost, and say this is like extremely rapid quenching, so the resulting solutions are usually metastable. Their alternative lets controlled uphill steps in, in cost terms, and so the procedure need not get stuck, since transitions out of a local optimum are always possible at nonzero temperature. In this post's terms, that is a controlled, deliberate step downhill.

A small simulation, this post's own, shows how big the effect can be. The landscape is the one sketched above: four peaks, the highest about 1.6 times as high as the next best. Starting from 1,000 random positions, a climber that only takes improving steps ended on the highest peak 26% of the time. That is just the share of the map whose slope leads there. Starting five climbers from random positions and keeping the best reached the highest peak 75% of the time. A simulated-annealing climber that accepts some downhill steps early and fewer later reached it 63% of the time (300 starts). The settings were not tuned, so the ranking of the two remedies should not be generalised. The point is that the plain climber is wrong about three times out of four on this landscape.

## The objective itself can mislead

The trap is not only in the method. In [a 2011 paper](https://pubmed.ncbi.nlm.nih.gov/20868264/), Joel Lehman and Kenneth Stanley argue that objective functions can actively misdirect search toward dead ends, a problem they call deception. They propose searching for novelty and ignoring the objective, and report that in maze navigation and biped walking tasks, novelty search significantly outperformed objective-based search. For a person, the equivalent is that the metric being improved can be the thing that keeps one on the small hill. A measure that has been rising for a year is evidence about that hill only.

## Signals that the summit is local

From inside a landscape the whole terrain is not visible, so the signals are indirect. These three are heuristics, not tests, and they are the post's own selection:

- **Each change gives less than the last.** Improvements are shrinking.
- **More input does not give more output.** Extra hours, people or money no longer move the result.
- **Someone with less skill is overtaking from an adjacent hill.** The gap is not ability, so it is position.

None of these separates a summit from a plateau. They only mark when to start looking at neighbouring hills, and looking is cheap compared with moving.

## Price the descent before taking it

The descent is the hard part. Going to a higher peak means a stretch of worse results, and its cost is visible and immediate while the gain is hidden and uncertain. This asymmetry is why many people stay. Two measures keep it from being decided in the moment.

First, put a number on the descent in advance: how much worse a result, for how long, is acceptable. A vague dread of loss decides differently from a stated limit. The previous post's habit of imagining the failure first applies here, with the failure being the valley.

Second, enter the valley partially. The annealing result points to a design: keep climbing the current hill, while a fixed slice of time, say 10 to 20 percent, goes to a different direction. This slice is the post's own analogy to the temperature in the annealing method, and the percentage is a judgment call, not a research result. The restart result in the simulation is the same idea in another form, with a cost: five starts took five times the climbing. The slice should be priced like any other hour, as in the [post on pricing time](/posts/price-your-own-hour/).

## The safeguard

The safeguard in this post is a decision made at the peak's foot instead of at its top: set the descent budget, name the signals that trigger a search, and fix the exploration slice, all before the good run ends. At the summit, every step downhill feels like a mistake. A rule written earlier is the only thing that can overrule that feeling.

The last post in the series turns to the largest delegation of thinking so far: what to hand to AI, and what to keep.

*Series: How to Think Clearly. 1. [Where Thinking Breaks and What to Put There](/posts/where-thinking-breaks-and-what-to-put-there/). 2. [A Thought Has a Machine Behind It](/posts/a-thought-has-a-machine-behind-it/). 3. [Catch the Error While It Happens](/posts/catch-the-error-while-it-happens/). 4. [Make the Right Thing the Easy Thing](/posts/make-the-right-thing-the-easy-thing/). 5. [The Map Is Not the Territory](/posts/the-map-is-not-the-territory/). 6. [Three Tools for Estimating the Unknown](/posts/three-tools-for-estimating-the-unknown/). 7. [Averages Lie](/posts/averages-lie/). 8. [Price Your Own Hour](/posts/price-your-own-hour/). 9. [Explore Early, Exploit Late, Imagine the Failure First](/posts/explore-early-exploit-late-imagine-the-failure-first/). 10. [The Hill You Are Standing On](/posts/the-hill-you-are-standing-on/). 11. [Let AI Think, Keep the Decision](/posts/let-ai-think-keep-the-decision/).*
