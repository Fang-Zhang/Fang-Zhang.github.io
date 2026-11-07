---
title: "Sailing by the Numbers (8/8): Autopilot and Weather Routing"
date: 2026-12-30 06:00:00 +1300
categories: [Marine]
tags: [sailing-by-the-numbers, sailing, racing]
image:
  path: /assets/pic/sailing-post-8.png
  alt: Eight stage boxes for the series, with stage 8, Autopilot and routing, highlighted
---

Two kinds of software now make decisions that a sailor once made. One acts every fraction of a second: the autopilot, which moves the rudder. The other acts every few hours: weather routing, which proposes a route. This post explains how each works, using public sources, and then ties the eight posts together.

**Where this sits:** stage 8 of 8, the last in the series.

![Eight stage boxes for the series, with stage 8 highlighted]({{ "/assets/pic/sailing-post-8.png" | relative_url }})
_Figure 1. The eight posts._

## The autopilot: a feedback loop

An autopilot compares a target with a measurement and moves the rudder to reduce the difference. [IMOCA's article on The Ocean Race](https://www.imoca.org/en/news/news/this-ocean-race-is-going-to-be-99-9-under-autopilot-but-nothing-has-been-lost-argues-sam-goodchild) describes the targets in their simplest form: a pilot can follow a compass course or a wind angle, and can be programmed to achieve a certain heel angle. On top of that, the article says, there are many details that can be applied to produce more precise steering, for example when a boat slaloms downwind in the Southern Ocean at 30 knots. The loop in Figure 2 is a generic sketch of the principle, not any maker's design.

![Boxes for target, autopilot, boat and sensors in a loop, with the crew adjusting the autopilot's settings]({{ "/assets/pic/sailing-autopilot-loop.png" | relative_url }})
_Figure 2. An autopilot as a feedback loop. A generic sketch._

The same article explains why a fast foiling boat leans on this loop. The sailor Sam Goodchild is quoted describing how, when a foil lifts the boat out of the water and it accelerates, the apparent wind angle might move 10 to 15 degrees, and after hitting a wave it might come back 20 degrees. A trimmer has to find an average setting that works, because it is impossible to ease two metres of traveller and mainsheet at every wave and winch it back on as the boat speeds up. That is the apparent-wind motion from the third post, at a speed where no person can follow it by hand.

The people do not leave the loop. Will Harris of Team Malizia is quoted in the same article: "The keypad is effectively our tiller now and it's not like we push the 'on' button and it goes off on its own." He says the settings are adjusted constantly and that, with someone untrained on the keypad, "the boat would be all over the place." So the work moves from steering to tuning the controller.

## Weather routing: choosing the route

Weather routing answers a different question: given a forecast, which route gets the boat to the finish fastest? A [SAIL magazine review of routing software](https://sailmagazine.com/cruising/weather-routing-101-part-2) describes the standard method. It is the isochrone method, first proposed in the 1950s. It uses a fixed time interval, and from the boat's position it branches out at many angles. For each branch, the forecast gives the wind, the boat's performance data gives the speed, and speed times time gives the distance. Repeating this step by step, the route that goes the farthest is the optimal route.

The inputs are the two things the earlier posts built: a forecast and a polar. The forecast usually comes as a GRIB file; [GRIB](https://en.wikipedia.org/wiki/GRIB) is a compact format for gridded meteorological data, approved by the World Meteorological Organization, with edition 2 approved in 2003. The polar is the one from the third post: the boat's speed for each wind speed and angle.

![A fan of branches from a start point, with lines for the positions reached after 3, 6 and 9 hours, and one route highlighted]({{ "/assets/pic/sailing-isochrone.png" | relative_url }})
_Figure 3. The isochrone method. The geometry is illustrative._

The article lists the options in the better programs: the isochrone interval, which defaults to three hours, limits on wind speed and wave height, adjustments to the polars, and, in most of them, tides and currents. Its author configured several programs with the same route, polar and forecast file and compared the results. The routes came out similar, especially during the first days of the forecast, and the differences came from default settings, waves and currents, and internal calculations. His conclusion: route optimisation is a tool to supplement the routing decision, and even the best computer model cannot replace common sense and judgment.

Two limits follow directly from the method, and they are my reading of it, not a claim from the article. The route is only as good as the forecast, and the forecast gets less reliable with each day. And it is only as good as the polar; a polar that is too optimistic, as the last post noted, will favour routes the boat cannot sail as fast as predicted.

## Who is allowed to use it

Whether a skipper may use shore-based routing depends on the race. In the Vendée Globe, as the sixth post showed, the notice of race prohibits receiving weather, tactical or strategic information useful for decision making that could improve performance or influence route choice. A skipper's own software on the boat, running on public forecast data, is a different matter from a shore team advising the skipper, and the rule is aimed at the second. I did not find the class's published position on routing software itself, so I do not claim one.

## The whole series in one chain

![Eight boxes in a row: races, rules, physics, sensors, kit, top teams, tactics, automation]({{ "/assets/pic/sailing-whole-chain.png" | relative_url }})
_Figure 4. The eight posts as one chain._

The first post asked who races and what they are trying to win. The rules decide who is right when boats meet and how a result is calculated. Physics decides where a boat can go: the wind triangle, the polar and the layline. Sensors turn that physics into numbers, and a network carries the numbers to a display. How much of that system a team may choose depends on its class rule. The public record shows the outline of what top teams do, not the settings. A crew can use the same numbers in a race by following a few old tactical rules and checking the log afterwards. And the last stage hands two jobs to software: steering moment by moment, and route choice hour by hour.

Each stage rests on a source that can be checked. Where there was none, the posts said so. That is the main way to read about sailing technology: ask which rule, which document or which measurement a claim comes from.

_Series: Sailing by the Numbers. 1. Who races, and for what. 2. The rules that decide who is right. 3. Why a boat goes where it goes. 4. Sensors and the data chain. 5. Kit by boat type. 6. What top teams use. 7. Racing with data. 8. Autopilot and weather routing (this post)._
