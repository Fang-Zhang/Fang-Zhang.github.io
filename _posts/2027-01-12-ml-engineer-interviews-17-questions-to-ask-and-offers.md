---
title: "ML Engineer Interviews, With Reference Answers (17/18): Questions to Ask, Warning Signs, Offers and a Preparation Method"
date: 2027-01-12 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [interview, negotiation, compensation, spaced-repetition, series]
description: "Part 17 of 18. The closing questions of an ML engineer interview loop: what to ask the interviewer, which warning signs have a source, how to compare offers with arithmetic, what research says about first offers, and how to schedule preparation with spacing and retrieval practice."
---

After the technical, project and behavioural rounds of [Part 16](/posts/ml-engineer-interviews-16-behavioural-and-values/) there are decisions on both sides, and a preparation schedule that ran for weeks. This post covers six questions of that stage: what to ask the interviewer, which warning signs are documented, how to compare two offers, what the research says about making the first offer, how to schedule revision, and how to learn from a rejection. Each has what it tests, a reference answer, a weak answer and a follow-up. The evidence is limited to what was read: Indeed's lists of [closing questions](https://www.indeed.com/career-advice/interviewing/good-questions-to-ask-at-the-end-of-an-interview) and [fraud signs](https://www.indeed.com/career-advice/interviewing/interview-scams), a [structured review of first-offer research](https://ncmr.lps.library.cmu.edu/article/id/574), a [Harvard Program on Negotiation article](https://www.pon.harvard.edu/daily/dealmaking-daily/resolving-the-first-offer-dilemma-in-business-negotiations), and two experimental papers on memory. The compensation figures are invented, and two scripts run on this machine (CPU, no randomness). Advice that no source supports is marked as the author's.

![Left: four-year total compensation of offer A, 848k, against offer B with its equity worth 0, half, as granted and 3 times, 605k, 725k, 845k and 1,325k. Right: assumed recall probability against test day for cramming, even spacing and doubling gaps; cramming falls to 0.075 by day 60 and spacing stays near 0.65.](/assets/pic/ml-interview-17-offers-and-spacing.png)
*Left: the offer comparison of question 3. Right: the spacing model of question 5, whose dynamics are assumed, not measured.*

## 1. "Do you have any questions for us?"

**What it tests.** Preparation, curiosity and what the candidate values. [Indeed](https://www.indeed.com/career-advice/interviewing/good-questions-to-ask-at-the-end-of-an-interview) advises preparing "at least two or three questions that express your interest, as well as show them that you've done your homework by researching the company". Its list groups questions by company (culture, values, challenges, competitors), by role (day-to-day responsibilities, reporting line, "Is this a new position? If not, why did the person before me leave this role?", "How do you evaluate success?", "What are your expectations for this role during the first 30 to 60 days?") and by the candidate's own standing ("Do you have any hesitations about my qualifications?").

**Reference answer.** Pick questions whose answers would change the decision, and fit each to the interviewer in front of you. The author's ML-specific versions of Indeed's categories:

- To the hiring manager: how is success of this role measured in the first six months, and who decides it? What was the last model that was shipped, retired or rolled back, and why?
- To a peer engineer: how long does it take from an idea to a model in production? How are offline metrics linked to the product metric (Part 11)? Who owns the data pipeline, and what happens when it breaks?
- To a senior leader: what is the team's plan if the main model is not good enough in a year? Which decisions does the team make alone, and which need approval?
- To anyone: is this a new position? If not, why did the previous person leave? (from Indeed's list)

Ask the question that follows from an answer you were given, not only the prepared one: it shows that you listened.

**Common weak answer.** "No, I think you covered everything", or questions answerable from the careers page, or a first question about holidays.

**Follow-up.** "What worries you most about this role?" This turns the candidate's own question to them: name a real concern from what was heard, for instance data ownership, and say how you would test it in the first month.

## 2. "How do you tell a bad opportunity from a good one?"

**What it tests.** Judgement about the employer, which candidates usually treat as the employer's job.

**Reference answer.** Separate documented warning signs from the author's judgement. Documented: [Indeed lists fifteen signs](https://www.indeed.com/career-advice/interviewing/interview-scams) that an opportunity is fraudulent, including a job that is too good to be true, no company information online, vague requirements, an interview held over instant chat, a request for money or for software to be paid for, a request for confidential information, an interviewer who avoids answering questions, and an immediate job offer. The list is about scams, and it applies to the early stage of a search where recruiters contact candidates unprompted; verify the company, the person and the domain before sharing documents. Not documented (the author's), signs inside a real company: answers to "how do you measure success?" that differ across interviewers; no one can say who owns the data or the model after launch; the role description changes between rounds; the team cannot name a model that was retired; the interviewers cannot describe the last project that failed. Each is a sign that the role is not defined or that problems are not discussed, and each can be tested by asking the same question of several people and comparing the answers (question 1).

**Common weak answer.** Treating any imperfection as a reason to decline, or ignoring all signs because the brand is well known.

**Follow-up.** "What would you do if three interviewers gave you three different definitions of the role?" Ask the hiring manager to reconcile them, and treat the reply as data.

## 3. "How do you compare two offers?"

**What it tests.** Ability to turn a messy package into numbers, state the assumptions and see where the uncertainty sits.

**Reference answer.** Write down every component by year, value each under explicit assumptions, and compare scenarios and not a single number. The script below compares two invented offers: A, with a public company's stock valued at its grant value, and B, a private company with a higher bonus target and a larger grant whose value is unknown. Equity vests 25% a year after a one-year cliff, and bonuses pay out at 80% of target.

```python
# Two hypothetical offers; all figures are invented for illustration. Equity vests 25% a year after a one-year cliff, paid as 4 equal annual tranches.
offers = {
  "A (public company)": dict(base=150_000, bonus_target=0.10, payout=0.8, equity_grant=200_000, equity_scenarios={"flat": 1.0}),
  "B (private company)": dict(base=135_000, bonus_target=0.15, payout=0.8, equity_grant=240_000, equity_scenarios={"worth 0": 0.0, "worth half": 0.5, "as granted": 1.0, "worth 3x": 3.0}),
}
def total(o, years, mult):
    cash = years * o["base"] * (1 + o["bonus_target"] * o["payout"]); vested = min(years, 4) // 1 * 0.25 if years >= 1 else 0.0   # nothing vests before the one-year cliff
    return cash + vested * o["equity_grant"] * mult
print("total compensation received (cash + vested equity at the stated value multiple)")
for name, o in offers.items():
    for sc, m in o["equity_scenarios"].items():
        print(f"{name:20s} equity {sc:11s} | leave before 1 yr: {total(o, 0.9, m):>9,.0f} | 2 yrs: {total(o, 2, m):>9,.0f} | 4 yrs: {total(o, 4, m):>9,.0f}")
a4 = total(offers["A (public company)"], 4, 1.0); b = offers["B (private company)"]
be = (a4 - 4 * b["base"] * (1 + b["bonus_target"] * b["payout"])) / b["equity_grant"]
print(f"over 4 years B's equity must be worth {be:.2f}x its grant value for B to match A's total ({a4:,.0f})")
print(f"effect of a successful ask of +5,000 base on A over 4 years, with the bonus: {4 * 5000 * (1 + 0.10 * 0.8):,.0f}")
```

```
total compensation received (cash + vested equity at the stated value multiple)
A (public company)   equity flat        | leave before 1 yr:   145,800 | 2 yrs:   424,000 | 4 yrs:   848,000
B (private company)  equity worth 0     | leave before 1 yr:   136,080 | 2 yrs:   302,400 | 4 yrs:   604,800
B (private company)  equity worth half  | leave before 1 yr:   136,080 | 2 yrs:   362,400 | 4 yrs:   724,800
B (private company)  equity as granted  | leave before 1 yr:   136,080 | 2 yrs:   422,400 | 4 yrs:   844,800
B (private company)  equity worth 3x    | leave before 1 yr:   136,080 | 2 yrs:   662,400 | 4 yrs: 1,324,800
over 4 years B's equity must be worth 1.01x its grant value for B to match A's total (848,000)
effect of a successful ask of +5,000 base on A over 4 years, with the bonus: 21,600
```

At the grant value, B's four-year total (844,800) is within 0.4% of A's (848,000) and the break-even multiple is 1.01 times. The choice therefore rests on what the private equity will be worth, which is the number no one in the comparison knows: at half the grant value B is 123,200 lower, and at three times it is 476,800 higher. Leaving before the first anniversary gives only cash in both offers (145,800 and 136,080), so the stay length is also an input. A successful request of 5,000 more base on A is worth 21,600 over four years including the bonus, small beside the equity uncertainty on B but certain. The attributes that do not fit into the arithmetic (the work, the team, the manager, growth, location) go in a separate column and are weighed after the numbers (the author's method). Tax treatment, option exercise costs and dilution are outside this model and differ by country and instrument; ask a qualified adviser for those.

**Common weak answer.** Comparing base salaries only, or valuing private-company equity at the grant figure without saying that it is an assumption.

**Follow-up.** "What would you ask the company about the equity?" The author's list: the number of shares and the total outstanding, the type (options or shares), the strike price and exercise window, the vesting schedule with the cliff, and what happens in a sale or if you leave.

## 4. "Should you make the first offer in a salary negotiation?"

**What it tests.** Whether the candidate knows what is established about negotiation and what is folklore.

**Reference answer.** [Lipp, Smolinski and Kesting](https://ncmr.lps.library.cmu.edu/article/id/574) reviewed 119 journal articles published since 1967 and open with the statement that "there is a wide consensus that first offers have a significant impact on negotiation outcomes by causing an anchoring effect". The same abstract also says that extant research contains "anecdotal advice for negotiators", that no holistic overview of the findings existed, and that crucial research gaps remain, including the interactive nature of real negotiations. The [Harvard Program on Negotiation](https://www.pon.harvard.edu/daily/dealmaking-daily/resolving-the-first-offer-dilemma-in-business-negotiations) summarises: making the first offer often yields better financial results, but it can also increase negotiators' anxiety and decrease their overall satisfaction; it advises making the first offer when the negotiator has a strong outside alternative, understands the zone of possible agreement and has practised managing nerves. For a candidate, the reference answer is therefore conditional, not a rule: the number mentioned first tends to become the reference point, so when you know the market range and have a competing offer or a good current position, state a number with reasons; when you do not know the range, find out first (compare peers, ask the recruiter for the band). Two practical points (the author's): state a number as a range whose lower end you would accept, and ask for the whole package, since the comparison in question 3 shows that a base-salary gain can be small next to equity.

**Common weak answer.** "Never say a number first" or "always say a number first", both stated as laws, and accepting the first offer in order to avoid conflict.

**Follow-up.** "The recruiter says the offer is final." Ask which parts are fixed, ask about the other components (sign-on, equity, review timing) and say when you will decide.

## 5. "How did you prepare, and how would you prepare for the next one?"

**What it tests.** Whether the candidate has a method rather than a pile of hours. Interviewers ask it, and candidates need it for themselves.

**Reference answer.** Two findings from experimental research bear on a revision plan. [Cepeda et al.](https://pubmed.ncbi.nlm.nih.gov/16719566) analysed 839 assessments of distributed practice in 317 experiments from 184 articles and found that the interval between study sessions and the retention interval operate jointly: "the ISI producing maximal retention increased as retention interval increased." [Roediger and Karpicke](https://pubmed.ncbi.nlm.nih.gov/16507066) gave students prose passages to study, then either tested them or let them restudy, and tested again after 5 minutes, 2 days or a week. After 5 minutes restudying gave better recall; on the delayed tests prior testing produced "substantially greater retention than studying", even though repeated studying increased the students' confidence in their ability to remember. So the plan (the author's application) is to test yourself, not reread, and to spread the tests out. The simulation shows what follows from one simple model of forgetting, with the assumption stated in the first line of its output: recall decays exponentially with a stability that grows after each review, and more when the review happens after more forgetting.

```python
import numpy as np
# Illustration under ASSUMED dynamics, not fitted to data: recall R = exp(-dt / S); a review at recall R raises S by the factor 1 + a * (1 - R).
a, S0 = 3.0, 1.5
def recall(days, test_day):
    S, last = S0, 0.0
    for d in sorted(days): R = np.exp(-(d - last) / S); S *= 1 + a * (1 - R); last = d
    return np.exp(-(test_day - last) / S)
plans = {"no review after first study": [], "10 reviews on day 0 (massed)": [0.0] * 10, "10 reviews in the last 2 days (cramming)": list(np.linspace(28, 29.9, 10)),
         "10 evenly spaced reviews, day 3 to 29": list(np.linspace(3, 29, 10)), "5 reviews with doubling gaps: days 1,3,7,15,29": [1, 3, 7, 15, 29]}
tests = [30, 37, 60]
print(f"probability of recalling a card (assumed model: S0 {S0} d, gain factor a = {a}); reviews end by day 30")
print(" " * 49 + "".join(f"test day {t:<3d}" for t in tests))
for k, v in plans.items(): print(f"{k:48s} " + "".join(f"{recall(v, t):<12.3f}" for t in tests))
```

```
probability of recalling a card (assumed model: S0 1.5 d, gain factor a = 3.0); reviews end by day 30
                                                 test day 30 test day 37 test day 60 
no review after first study                      0.000       0.000       0.000       
10 reviews on day 0 (massed)                     0.000       0.000       0.000       
10 reviews in the last 2 days (cramming)         0.991       0.543       0.075       
10 evenly spaced reviews, day 3 to 29            0.987       0.903       0.673       
5 reviews with doubling gaps: days 1,3,7,15,29   0.986       0.895       0.651
```

The numbers are consequences of the assumed dynamics, not evidence about people. They illustrate two qualitative points that the cited studies support: cramming works for a test that comes immediately afterwards (0.991 on day 30, level with spacing at 0.987) and fails for a later one (0.075 on day 60 against 0.673), and five reviews with doubling gaps give almost the same result as ten evenly spaced ones (0.651 against 0.673 on day 60). For a loop that spans weeks, with several rounds on different days, that means the technical material must be reviewed on a spacing schedule well before the first round, and the final days spent on practice questions under time limits (retrieval) instead of rereading. Practically, keep a list of questions as cards (one per question in this series), answer without notes and write the answer down, mark the ones you missed, and bring them back at growing intervals.

**Common weak answer.** "I solved 300 problems" or a count of hours, with no mention of what was reviewed when, or of how errors were handled.

**Follow-up.** "What did you stop doing?" The method should have removed something, for instance rereading solutions.

## 6. "You were rejected. What did you learn?"

**What it tests.** Whether the candidate treats the outcome as information. It matters to the interviewer when the candidate is asked about earlier searches, and to the candidate for the next round.

**Reference answer.** A single rejection carries little information, because the interview is a noisy measurement: Part 16 gave the validity of structured interviews as 0.42 (Sackett et al.), meaning most of the variation in later performance is not captured by the score. The useful record is per question, not per company: after each interview, write the questions asked, what was said, where you were uncertain and the correct answer afterwards, and feed the uncertain ones back to the schedule of question 5 (the author's method). If feedback is offered, ask for it and note what it covers; if it is not, do not infer a cause from silence. Over several interviews the pattern appears in the log: the same category of missed question points to a topic, and a mixed pattern points to variance.

**Common weak answer.** "They did not see my potential", or a rewritten account in which the interviewer was at fault.

**Follow-up.** "What did you change after the third rejection?" The answer is a change in the log and the schedule, with the topic named.

## Takeaways

- Prepare two or three closing questions whose answers would change your decision, and ask them of more than one person.
- Indeed documents fifteen signs of a fraudulent opportunity; signs inside real companies (differing definitions of success, no owner for data, no retired models) are the author's heuristics and are tested by comparing answers.
- Compare offers as scenarios: in the example the equity value decided the result, with a break-even of 1.01 times the grant value, a 123,200 gap at half and a 476,800 gap at three times.
- First offers anchor outcomes according to a review of 119 articles, but making one can raise anxiety and lower satisfaction; the advice is conditional on knowing your alternatives and the range.
- Test yourself instead of rereading, and space the tests: the recall curves cross over after the test day, not at it.
- Keep a per-question interview log; a rejection is a noisy measurement.

## Limits

- The offers are invented, with an assumed bonus payout and an equity schedule; they omit tax, option exercise, dilution, benefits and location, which can change the comparison.
- The negotiation sources are a structured review and a university programme's article; the review itself says that the field lacks a complete picture of the interaction. Nothing here is a guarantee about salary outcomes, and norms differ by country and company.
- The forgetting model is an assumption, with parameters (an initial stability of 1.5 days and a gain factor of 3) that were not fitted to data. The two memory papers were read at the level of their abstracts, and their tasks were verbal recall and prose passages, not interview skills.
- The fraud signs come from a single job-site article about scams, and the in-company signs, the question lists for ML roles, the log method and the negotiation points marked as the author's have no source behind them.
- Not covered: visas and relocation, counter-offers from a current employer, contract terms, and equity in each country's tax regime.

Next: [Part 18, further reading for every round](/posts/ml-engineer-interviews-18-further-reading/).
