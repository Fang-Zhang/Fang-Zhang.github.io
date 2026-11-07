---
title: "Sailing by the Numbers (3/8): Why a Boat Goes Where It Goes"
date: 2026-12-20 06:00:00 +1300
categories: [Marine]
tags: [sailing-by-the-numbers, sailing, racing]
image:
  path: /assets/pic/sailing-post-3.png
  alt: Eight stage boxes for the series, with stage 3, Why boats go where they go, highlighted
---

A sailboat cannot sail straight into the wind, and the fastest way to a mark is rarely a straight line toward it. This post explains why, using four ideas: apparent wind, the polar diagram, velocity made good (VMG), and what changes when a boat is fast enough to fly on foils. It is the physics behind every instrument later in the series.

**Where this sits:** stage 3 of 8. The first two posts covered the races and the rules. From here on the posts are about measurement. All numbers in the figures are invented to show the shape of the idea, and none belongs to a real boat.

![Eight stage boxes for the series, with stage 3 highlighted]({{ "/assets/pic/sailing-post-3.png" | relative_url }})
_Figure 1. The eight posts._

## Two winds

Sailors distinguish two winds. The **true wind** is the wind over the water, the one a flag on a buoy would show. The **apparent wind** is the wind felt on the moving boat, the one a masthead wind vane measures. The [US Sailing guide to ORC polars](https://www.ussailing.org/wp-content/uploads/2018/01/Speed-Guide-Explanation.pdf) notes that it is the apparent wind that is felt or observed on the boat, and gives the conversion in its Appendix A. In vector terms, the apparent wind is the true wind minus the boat's own velocity. Moving forward adds a headwind to whatever the air is doing.

Take a 10-knot true wind blowing across the boat from the side. A boat doing 6 knots feels 11.7 knots, coming from 59 degrees off the bow. A boat doing 30 knots in the same air feels 31.6 knots from 18 degrees off the bow. The air itself has not changed, but the wind on board is three times stronger and has swung almost to the front.

![Two vector triangles: at 6 knots the apparent wind is 11.7 knots from 59 degrees; at 30 knots it is 31.6 knots from 18 degrees]({{ "/assets/pic/sailing-wind-triangle.png" | relative_url }})
_Figure 2. The same 10-knot true wind, as felt by a slow boat and a fast one. The arithmetic is mine: apparent speed = sqrt(10² + boat speed²) when the true wind is abeam._

Sails are trimmed to the apparent wind, because that is the airflow they actually sit in. Racing crews still care about the true wind, because it describes the weather and the course, and the shift between the two is what instruments compute.

## The polar diagram

How fast a boat sails depends on the wind strength and on the angle between the wind and her course, the **true wind angle** (TWA). Plotting boat speed against angle for one wind strength gives a curve called a **polar diagram**. The [US Sailing guide](https://www.ussailing.org/wp-content/uploads/2018/01/Speed-Guide-Explanation.pdf) describes the standard layout: the wind angle runs from 0 at the top to 180 at the bottom, and the farther a point lies from the centre along an angle, the higher the predicted speed. It also describes the curves it plots, for six wind speeds from 6 to 20 knots, and points out that close to the wind the speed hardly rises with stronger wind, while reaching speeds rise a lot.

Where do the numbers come from? For rating purposes, from a computer model. The guide says the ORC's velocity prediction program (VPP) simulates the boat from hull measurements and from research on hulls in test tanks and sails in wind tunnels, and finds the speed at which drag balances the drive of the sails; the [ORC VPP page](https://orc.org/for-designers/orc-vpp) describes the same program. Crews then amend those theoretical polars with measurements from their own boat, which the guide also recommends.

![A half polar diagram: boat speed against true wind angle, with the best upwind angle near 43 degrees and the best downwind angle near 150 degrees marked]({{ "/assets/pic/sailing-polar.png" | relative_url }})
_Figure 3. An invented polar diagram for a 10-knot wind. The shape, narrow near the wind and widest at the side, is typical; the values are not._

## Velocity made good

A boat's speed through the water is not the same as her progress toward the next mark. **Velocity made good** (VMG) is the part of her speed that points at, or away from, the wind. Upwind, it is boat speed multiplied by the cosine of the true wind angle. The guide calls the best sailing angles for VMG toward a windward or leeward mark the most valuable information a polar provides.

The trade-off is visible in the invented polar. Sailing 30 degrees off the wind is slow, 3.0 knots toward the mark. Sailing 60 degrees off is faster through the water but points less at the mark, 3.4 knots. In between, at about 43 degrees, the product peaks at 4.1 knots. Sailing straight at the mark is not an option, so the boat zigzags, and the best zigzag uses that angle.

![Left: a zigzag course to a windward mark at the best angle, avoiding the sector into the wind. Right: curve of speed toward the wind against true wind angle, peaking near 43 degrees]({{ "/assets/pic/sailing-vmg.png" | relative_url }})
_Figure 4. Upwind VMG for the invented polar in Figure 3._

The same reasoning applies downwind, and the result can surprise a newcomer. With this polar, sailing dead downwind at 180 degrees gives 4.3 knots toward the mark. Sailing at about 150 degrees gives 5.6 knots through the water and 4.8 knots toward the mark, so the boat does better by zigzagging downwind too, gybing from side to side. The [US Sailing guide](https://www.ussailing.org/wp-content/uploads/2018/01/Speed-Guide-Explanation.pdf) says the optimum angles for windward-leeward courses are marked on its polar diagrams, and it shows how the angle on one tack tells you the course on the other. The 150-degree figure here is a property of my invented curve, not a rule: real boats differ, and foiling boats differ most.

## When the boat outruns the wind

The effects above are small at low speed and dominant at high speed. The [America's Cup's own technical article](https://www.americascup.com/news/4388_INSIDE-THE-AC75-HOW-THE-DOUBLE-SKINNED-MAINSAIL-REALLY-WORKS) on the AC75 says the boat's speed can be several times the true wind speed, so that small changes in speed move the apparent wind direction a lot. Once the boat is travelling at 40-plus knots, it says, the apparent wind has moved dramatically forward, and the sail must be flattened and depowered because drag becomes increasingly expensive.

The article also describes the strangest consequence. An AC75 sailing toward the downwind gate is still experiencing apparent wind from well forward, and the rig works much more like a yacht reaching or sailing close to the apparent wind. Her sails never look like the deeply eased sails of a conventional boat running downwind, because the boat's own velocity creates the wind. Figure 2 shows the mechanism: the faster the boat, the smaller the apparent wind angle, whatever the true wind does.

For a foiling boat the question "which angle is fastest to the mark" therefore has a different answer from a slow boat's, and it changes continuously with speed. A simple rule of thumb written on a card is not enough. The boat needs to know her true wind, her apparent wind and her speed, accurately and continuously, and that is the job of the sensors in the next post.

## What this means for the numbers

Three quantities carry most of the load: boat speed, apparent wind and true wind. Everything else a tactician quotes, such as target angles, VMG and the distance to a layline, is computed from them. The polar is the reference that says whether the numbers on the screen are good. The rest of this series is about how those numbers are measured, how far they can be trusted, and how crews use them.

_Series: Sailing by the Numbers. 1. [Who races, and for what](/posts/sailing-by-the-numbers-1-who-races/). 2. [The rules that decide who is right](/posts/sailing-by-the-numbers-2-the-rules/). 3. Why a boat goes where it goes (this post). 4. [Sensors and the data chain](/posts/sailing-by-the-numbers-4-sensors-and-the-data-chain/). 5. [Kit by boat type](/posts/sailing-by-the-numbers-5-kit-by-boat-type/). 6. [What top teams use](/posts/sailing-by-the-numbers-6-what-top-teams-use/). 7. [Racing with data](/posts/sailing-by-the-numbers-7-racing-with-data/). 8. [Autopilot and weather routing](/posts/sailing-by-the-numbers-8-autopilot-and-weather-routing/)._
