---
title: "Fishing with Sonar"
date: 2026-09-24 07:30:00 +1200
categories: [Personal, Day Skipper]
tags: [sailing, navigation, sonar, fishing]
description: "How echo sounders actually work, and how the same instrument the course teaches for depth safety doubles as a fish-finder on the water."
---

Part of the [Day Skipper Notes]({{ '/posts/day-skipper-notes/' | relative_url }}) series.

The Day Skipper course teaches the echo sounder purely as a safety instrument — keeping you off the bottom. What I found interesting is that the exact same principle, with a slightly different interpretation lens, is what makes it useful for fishing. Same hardware, same physics, two very different questions being asked of it.

## The physics: it's just an echo

An echo sounder (depth sounder / sonar) sends a short pulse of sound down through the water from a transducer mounted in the hull, and listens for the reflection. Sound travels through seawater at a fairly stable ~1,500 m/s, so the time between sending the pulse and hearing the echo converts directly into distance: depth equals (speed of sound × time) divided by two, since the pulse travels down and the echo travels back.

The frequency matters. Lower frequencies (around 50 kHz) penetrate deeper but return a coarser picture; higher frequencies (200 kHz) give sharper detail but attenuate faster and are better suited to shallower water. Fishfinder-oriented sonar units often run both simultaneously and show you two overlapping pictures for exactly this reason.

## Reading depth for navigation

For pure navigation safety, you care about one number: the depth right now, compared against your charted depth plus tidal height, to confirm you're where you think you are (or to catch you *before* you're somewhere you shouldn't be). The course drills a specific habit: always know your echo sounder's "keel offset" setting — whether it's reporting depth from the transducer, from the waterline, or from the keel — because a few tenths of a metre of misconfiguration is exactly the margin that matters when you're threading a shallow channel at low tide.

## Reading the same trace for fish

A fishfinder is reading the same echo, but instead of reducing it to a single depth number, it keeps the full strength and shape of the return and displays it as a scrolling waterfall image. Fish don't reflect sound the way a hard, flat seabed does — a fish's swim bladder (the gas-filled sac that gives it buoyancy) is a strong, distinct reflector, different in shape and intensity from the seabed line underneath it. On screen this shows up as a discrete arch or blob suspended above the bottom trace, rather than merging into it.

The practical skill is learning to tell apart: a hard, thin seabed line (rock or sand), a soft, thick seabed line (mud, weed), bait balls (a dense, textured cloud, often mid-water), and individual fish arches (crisp, isolated marks). None of this changes the underlying instrument — it changes what you're trained to notice in an otherwise identical stream of pulses and echoes.

## Where the two uses genuinely diverge

Navigation-focused depth sounders prioritize a stable, averaged number you can trust at a glance — noise gets smoothed out. Fishfinder-focused sonar wants sensitivity to exactly the kind of small, transient returns that navigation smoothing would filter away as noise. A unit tuned hard for one job is usually mediocre at the other, which is why serious fishing sonar and a boat's primary safety depth sounder are often separate transducers even when they live in the same display unit.

## The habit that carries over either way

Whichever question you're asking of the echo — "is this safe to sail over" or "is there anything worth casting at down there" — the instrument is only as good as your trust in its calibration and offset settings. The Day Skipper habit of checking and knowing your sounder's settings before relying on it is the same habit that makes it useful for fishing, just pointed at a different kind of return.
