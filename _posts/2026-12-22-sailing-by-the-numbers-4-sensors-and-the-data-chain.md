---
title: "Sailing by the Numbers (4/8): Sensors and the Data Chain"
date: 2026-12-22 06:00:00 +1300
categories: [Marine]
tags: [sailing-by-the-numbers, sailing, racing]
image:
  path: /assets/pic/sailing-post-4.png
  alt: Eight stage boxes for the series, with stage 4, Sensors and data, highlighted
---

Every number on a racing boat's screen is either measured by a sensor or calculated from measurements. This post follows that chain: which quantities are measured, how they travel around the boat, which are calculated, and why a small error at the start of the chain turns into a larger one at the end. It treats the systems generically. Real products differ in detail, and the figures are self-drawn mock-ups rather than any maker's screens.

**Where this sits:** stage 4 of 8. The previous post explained apparent wind, polars and VMG. This one covers how a boat gets those quantities.

![Eight stage boxes for the series, with stage 4 highlighted]({{ "/assets/pic/sailing-post-4.png" | relative_url }})
_Figure 1. The eight posts._

## What is measured

Four sensors supply almost everything:

- **Wind sensor** at the masthead: apparent wind angle and apparent wind speed.
- **Speed sensor** in the hull: speed through the water.
- **Compass:** heading.
- **Satellite receiver:** position, plus speed and course over the ground.

The difference between the two speeds matters. Speed through the water is how fast the water is passing the hull, which is what the sails and the polar care about. Speed over the ground is how fast the boat moves across the seabed, which is what the course to a mark cares about. They differ by the current.

Two of the sensors are worth a sentence each. An ultrasonic wind sensor [measures the time sound pulses take to travel between fixed transducers](https://gillinstruments.com/knowledge/how-does-an-ultrasonic-wind-sensor-work): a pulse travelling with the wind arrives sooner than one travelling against it, and the differences across several paths give wind speed and direction, with no rotating cups, vanes or bearings. A common speed sensor is a small [paddlewheel mounted in the hull](https://www.chicagomarineelectronics.com/Airmar_Transducers_Sensors.htm), turned by the passing water.

## How the data travels

Older instruments passed data over point-to-point wires using [NMEA 0183](https://actisense.com/news/retrofitting-nmea-2000-legacy-nmea-0183-equipment), which sends human-readable text sentences, usually at 4800 baud. If a satellite receiver has to talk to a chartplotter, an autopilot and a radio at once, each connection needs its own wiring.

[NMEA 2000](https://www.nmea.org/nmea-2000.html) replaced that with a network. The standard is based on CAN (Controller Area Network) and describes, in NMEA's words, a low-cost, moderate-capacity, bi-directional, multi-transmitter and multi-receiver instrument network. Every device plugs into a single backbone cable through a short drop and a T-piece; every device can transmit and receive. The data runs at [250 kbit/s](https://actisense.com/news/retrofitting-nmea-2000-legacy-nmea-0183-equipment) as compact binary messages labelled by Parameter Group Number (PGN), so a new device added anywhere on the backbone becomes part of the network. A maker's [guide to the standard](https://actisense.com/wp-content/uploads/2021/08/Complete-Guide-to-NMEA-2000_Revision_4_2021.pdf) gives the layout limits: a drop is at most 6 m, all drops together at most 78 m, and the backbone 100 m to 250 m depending on the cable type.

![A horizontal backbone with terminators at both ends and seven devices hanging off T-pieces on short drops]({{ "/assets/pic/sailing-nmea-topology.png" | relative_url }})
_Figure 2. The NMEA 2000 layout: one backbone, a terminator at each end, one short drop per device._

## What is calculated

Wind sensors measure the wind the boat feels. The true wind, the boat's VMG and the target speed from the polar are all derived. The conversion is the one from the previous post: the processor takes the measured apparent wind, subtracts the boat's velocity, and gets the true wind, using the formulas published in the [US Sailing guide to ORC polars](https://www.ussailing.org/wp-content/uploads/2018/01/Speed-Guide-Explanation.pdf). With the true wind angle and speed it looks up the polar for the target speed and angle, and computes the VMG. A tactical display also shows distances such as the distance to a layline, the line along which a boat can just reach a mark on one tack.

![Four sensors feed a network, which feeds a processor that calculates true wind, VMG, target speed and layline distance, which feeds displays, autopilot and logging]({{ "/assets/pic/sailing-data-chain.png" | relative_url }})
_Figure 3. From sensors to the numbers on the screen. A generic diagram._

![A generic dark instrument screen with eight tiles: apparent wind angle 28 degrees, apparent wind speed 14.6 knots, true wind angle 43 degrees, true wind speed 10 knots, speed through water 5.5 knots, VMG 4.0 knots, target speed 5.6 knots, heading 312 degrees]({{ "/assets/pic/sailing-generic-display.png" | relative_url }})
_Figure 4. A self-drawn generic display. The values continue the invented example from the previous post: 10 knots of true wind, a boat sailing at 43 degrees to it, slightly below the polar target._

Read the screen as a crew would. The boat is sailing 5.5 knots against a target of 5.6, about 98 percent of the polar. The VMG of 4.0 knots follows from 5.5 multiplied by the cosine of 43 degrees. Four of the tiles are measured directly: the apparent wind angle and speed, the speed through the water, and the heading. The other four are calculated from those.

## Why calibration matters

Because the true wind is calculated, an error in a measurement reaches every quantity downstream. A worked example with the numbers above: suppose the speed sensor reads 10 percent low, 5.0 knots instead of 5.6. The apparent wind has been measured correctly, so the processor subtracts a boat speed that is too small. It reports a true wind angle of 41 degrees instead of 43, and a true wind speed of 10.4 knots instead of 10.0. The boat appears to point higher than she does and to be in more wind than there is, and the target speed read from the polar is wrong too. The arithmetic is mine, from the vector formulas above.

Crews therefore calibrate. One instrument maker's [calibration guide](https://ockam.com/wp-content/uploads/2024/01/TDavis-Cal-Crib-Sheet.pdf) describes the standard method for the speed sensor: motor back and forth over a known distance while logging distance, and average the runs, which removes the effect of current. The calibration factor is then adjusted by the percentage difference between the logged and the known distance. The same guide treats wind angle errors, which it says generally result from the masthead unit not pointing straight, and *upwash*: the wind bends around the sail plan before it reaches the sails, like water around a rock in a river, so the masthead unit sees a different direction from the true wind further upstream. The guide says the effect is large when beating, smaller on a reach, and hardly present on a run, and larger with fuller sails and in light air, so it is found by sailing: tack the boat and watch the displayed wind direction.

A wind angle offset has a visible symptom. A masthead unit that is a few degrees off the centreline makes the angle read wider on one tack and narrower on the other. The same guide gives the fix: find the difference between the two tacks and apply half of it, because the correction takes away from one side and adds to the other. The check is simple: sail upwind on both tacks and compare.

## What this means for the numbers

The chain has three links: sensors, network and calculation. The network is usually the most reliable one. The sensors and the calibration are where most of the error enters, and the calculation passes it on without comment. A screen shows a decimal place of precision that the sensors behind it may not have. The next post looks at the kit that fills these roles on different types of boat, from club racers to foiling catamarans.

_Series: Sailing by the Numbers. 1. Who races, and for what. 2. The rules that decide who is right. 3. Why a boat goes where it goes. 4. Sensors and the data chain (this post). 5. Kit by boat type. 6. What top teams use. 7. Racing with data. 8. Autopilot and weather routing._
