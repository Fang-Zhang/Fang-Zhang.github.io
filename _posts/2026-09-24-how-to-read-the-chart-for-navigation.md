---
title: "How to Read the Chart for Navigation"
date: 2026-09-24 07:15:00 +1200
categories: [Personal, Day Skipper]
tags: [sailing, navigation, chartwork]
description: "What a nautical chart is actually encoding, and how to turn chart symbols into a position and a course you can trust."
---

Part of the [Day Skipper Notes]({{ '/posts/day-skipper-notes/' | relative_url }}) series.

Chartwork was the part of Day Skipper that felt most like a new language rather than a new skill. The mechanics — parallel rules, dividers, plotting a fix — are simple once shown. What took longer was understanding what a chart is actually a picture *of*.

## A chart is a projection, not a photograph

A nautical chart shows depth (soundings), the seabed type, hazards, buoyage, and land features, all referenced to a chart datum — usually the level of Lowest Astronomical Tide, meaning the depths printed are closer to the worst case than the average. Any depth you read off a chart is a floor, and the tide adds to it, not the other way around. That one fact reframes almost every depth-related decision on the water.

Charts also use a Mercator-style projection, which is why latitude scales on the side of the chart are not uniform — and critically, why you measure distance using the latitude scale next to your position, never the longitude scale at the top or bottom. Get that backwards and every distance you plot is wrong by an amount that changes with latitude.

## Position: fixing where you actually are

A "fix" is your actual position, established independently of where you assumed you'd be. Three ways this gets done in the course:

- **GPS fix** — trivial now, but the syllabus insists you can do without it, because it insists you understand *why* your other methods work.
- **Visual fix** — taking compass bearings on two or three known charted objects (a headland, a lighthouse, a beacon), plotting each as a line of position, and taking the intersection as your fix. Three bearings crossing at a point (or forming a small "cocked hat" triangle) gives you confidence in the fix; a big triangle means something's off — dodgy compass, misidentified object, or a bearing taken too slowly while the boat moved.
- **Dead reckoning (DR)** — projecting forward from your last known fix using course steered and speed through the water, without accounting for tide or current. It's your best estimate absent new information, and it's also explicitly a decaying estimate — the longer since your last real fix, the less you should trust your DR position.

## Estimated position vs dead reckoning

This distinction is easy to blur and the course is strict about it: DR only uses course and speed. An **estimated position (EP)** takes the DR position and applies your best estimate of tidal stream and leeway (the boat's sideways drift from wind pressure on the hull and rig). The EP is your actual best guess of where you are when you can't get a fix; the DR is one deliberately incomplete input into it.

## Plotting: what the pencil work is actually for

The physical act of plotting — parallel rule off the compass rose, dividers on the latitude scale — isn't busywork, it's forcing you to be explicit about assumptions that are easy to skip in your head: what course did I actually steer, what's my speed through the water, how long has it been, what's the tide doing right now. Digital chartplotters do this arithmetic invisibly, which is exactly why the course teaches it by hand first — so that when the electronics disagree with your gut, you have an independent way to check which one is wrong.

## Why this comes before radar and pilotage

Everything after chartwork — pilotage into a harbour, using radar to confirm a position, planning around tidal gates — depends on being able to translate what you see (or what an instrument reports) into a position on the chart you can reason about. Chartwork isn't a separate skill from those; it's the coordinate system all of them report back into.
