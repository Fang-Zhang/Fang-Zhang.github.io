---
title: "Sailing by the Numbers (7/8): Racing with Data"
date: 2026-12-28 06:00:00 +1300
categories: [Marine]
tags: [sailing-by-the-numbers, sailing, racing]
image:
  path: /assets/pic/sailing-post-7.png
  alt: Eight stage boxes for the series, with stage 7, Racing with data, highlighted
---

The earlier posts built the numbers: true wind, polars, VMG, laylines. This post puts them to work in one race, in the order a race happens: before the start, upwind, downwind, and afterwards. The tactical rules come from published racing guides. The start-line geometry and the post-race method are my own, and are marked as such.

**Where this sits:** stage 7 of 8. The first six posts described the system. This one is about using it.

![Eight stage boxes for the series, with stage 7 highlighted]({{ "/assets/pic/sailing-post-7.png" | relative_url }})
_Figure 1. The eight posts._

## Before the start: a median heading and a line

**The median heading.** David Dellenbaugh's Sailing World piece, [Windshift rules of thumb](https://www.sailingworld.com/how-to/windshift-rules-of-thumb), describes a pre-race routine that needs only a compass. Sail as long as possible on each tack and note the high and low headings. If the headings on starboard run from 240 to 270 degrees, the median heading is the average, 255. During the beat, compare the current heading with the median: at 245 the boat is 10 degrees below it, which means it is being headed, so tack to port; at 260 it is slightly lifted, so stay on starboard.

![Compass heading on starboard tack against time, oscillating between 240 and 270 with the median 255; above the median is lifted, below is headed]({{ "/assets/pic/sailing-median-heading.png" | relative_url }})
_Figure 2. The median-heading method, using the example numbers from the article. The curve is an illustration I drew, not a measurement._

**The start line.** A start line is almost never exactly square to the wind. If the line makes a bias angle with the square position, one end is further upwind. The difference in distance is the line length times the sine of the bias angle. As an illustration, a 200 m line with a 10-degree bias has one end about 35 m further upwind than the other. The calculation is mine. A US Sailing racing tips page adds a rule of thumb for shifty conditions: the middle of the line is usually a good home base, until one end is more than 15 degrees favoured.

![A start line tilted at the bias angle to a line square to the wind, with the upwind distance d marked]({{ "/assets/pic/sailing-line-bias.png" | relative_url }})
_Figure 3. Start line bias. The formula is geometry, not a rule from a source._

## Upwind: decide which kind of wind you have

The same Sailing World article puts the first decision plainly: is the wind oscillating, or shifting persistently in one direction? The strategy is different in each case.

![Two panels of wind direction against time: one swings back and forth, the other drifts in one direction]({{ "/assets/pic/sailing-oscillating-persistent.png" | relative_url }})
_Figure 4. The two cases, as illustrative traces._

In an oscillating breeze the article's rules of thumb are: tack on the headers, to stay on the lifted tack; keep off the laylines, because a lift there makes you overstand and a header leaves little room to tack; and when unsure which tack is lifted, choose the one on which the bow points closer to the mark. In a persistent shift the advice reverses: sail toward the shift, and keep sailing deeper into the header, since tacking at every header gives up the gain. One more rule connects the two: near the windward mark, play even an oscillating wind as a persistent shift. If an oscillation takes about four minutes, then within four minutes of the mark only one shift remains, and the boat will be round the mark before it swings back.

Another piece of numbers, from David Dellenbaugh's column on [US Sailing's racing tips page](https://www.ussailing.org/education/adult/certification-courses-endorsements/racing-tips-resources): in light air a three-knot increase in wind speed may improve boat speed by 30 to 40 per cent, while in heavy air the same increase may add only 5 to 10 per cent. That is why pressure, meaning where the stronger wind is, matters more in light air than in a breeze.

## Downwind: the same ideas, reversed

The article's downwind rules mirror the upwind ones. Upwind you tack on the headers; downwind you jibe on the lifts, which keeps the boat on the headed tack as long as possible. Stay away from the downwind laylines. Sail toward the puffs, since a good blast of air lets the boat sail lower and faster. And take the first jibe away from a persistent shift, the opposite of the upwind rule, unless the shift fills in stronger on one side, in which case head for the stronger wind.

## After the race: compare against the target

This section is my own method, not a source's. The polars from the third post give a target boat speed for each true wind speed and angle. A logged race gives the actual boat speed and the true wind at every moment, so the target can be looked up for each moment. Dividing the actual speed by the target speed gives a percentage of target for each leg. Three questions follow, in order of how cleanly the data can answer them:

1. **Speed.** On which legs, and at which wind angles, was the boat below target? That points to trim or steering.
2. **Manoeuvres.** How much speed was lost in each tack, and how long did it take to recover to target? Tacking on every header costs more than it gains if each tack is slow.
3. **Tactics.** After the race, did the headers and lifts the boat acted on match the median-heading rule? The log shows the wind direction; the crew knows what it was thinking.

The weakness of the method is the target itself. A polar from a rating certificate is a model, and the US Sailing guide cited in the earlier posts recommends amending it from the boat's own measurements. A boat that is always at 95 per cent of a model polar may simply have a polar that is too optimistic. Use the percentage to compare one leg with another, not to judge the crew by a single figure.

## What this means for the numbers

The tactical rules in this post are older than the instruments. The median heading needs a compass, and the rules of thumb date from the 1980s. What the data adds is a cheaper way to see the wind, a record to check against afterwards, and a way to tell speed problems from tactical ones. The last post covers the other automatic part of modern racing: the autopilot and weather routing.

_Series: Sailing by the Numbers. 1. [Who races, and for what](/posts/sailing-by-the-numbers-1-who-races/). 2. [The rules that decide who is right](/posts/sailing-by-the-numbers-2-the-rules/). 3. [Why a boat goes where it goes](/posts/sailing-by-the-numbers-3-why-boats-go-where-they-do/). 4. [Sensors and the data chain](/posts/sailing-by-the-numbers-4-sensors-and-the-data-chain/). 5. [Kit by boat type](/posts/sailing-by-the-numbers-5-kit-by-boat-type/). 6. [What top teams use](/posts/sailing-by-the-numbers-6-what-top-teams-use/). 7. Racing with data (this post). 8. [Autopilot and weather routing](/posts/sailing-by-the-numbers-8-autopilot-and-weather-routing/)._
