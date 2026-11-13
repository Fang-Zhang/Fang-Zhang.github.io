---
title: "Deep Learning, Domain by Domain, Part 8: Medicine, Biology and Drug Discovery"
date: 2027-02-04 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [deep-learning, medical-imaging, protein-structure, drug-discovery, series]
description: "Part 8 of Deep Learning, Domain by Domain. Eight papers in medicine, protein structure and drug discovery, read by one question: what was each result checked against?"
---

This post does one thing: it covers deep learning in medicine, biology and drug discovery through eight papers, and reads them by one question: what was the model's output checked against? In games the answer was a win; here it is a panel of clinicians, a laboratory experiment, or a measured structure. The image models are from [Part 3](/posts/deep-learning-domain-by-domain-3-images-and-video/) and the language models from [Part 4](/posts/deep-learning-domain-by-domain-4-language-and-speech/).

## The field: results that need an outside check

A model that classifies a retinal photograph has no built-in score. Someone has to decide what the correct answer is, and the papers below differ in who or what that is. The figure lists seven of the eight, with the reference each was measured against.

![A table of seven medical and biological papers, with columns for paper, input, what the output was checked against, and the reported result. Checked against: ISBI challenge benchmarks; a majority decision of ophthalmologists; 21 dermatologists on biopsy-proven cases; 4 radiologists; experimental protein structures; laboratory and mouse tests; clinicians' evaluation.]({{ "/assets/pic/dl-domain-8-checked-against.png" | relative_url }})
*What each paper's output was checked against, with the result each reports. The rows paraphrase each source's abstract; the 'checked against' column and its emphasis are the author's.*

## Medical imaging

**Microscopy (2015).** Ronneberger, Fischer and Brox's [U-Net](https://arxiv.org/abs/1505.04597) starts from the point that training deep networks was believed to need many thousand annotated samples. It uses a contracting path to capture context and a symmetric expanding path for precise localisation, and relies on data augmentation to use the few available annotated images more efficiently. The abstract reports that it outperforms the prior best method on the ISBI challenge for segmenting neuronal structures in electron microscopy stacks, wins the 2015 ISBI cell tracking challenge in two categories, and segments a 512x512 image in under a second on a recent GPU.

**Diabetic retinopathy (2016).** Gulshan and colleagues' [JAMA paper](https://jamanetwork.com/journals/jama/fullarticle/2588763) trained a convolutional network on 128,175 retinal images, each graded 3 to 7 times by a panel of 54 US licensed ophthalmologists and senior residents. It was validated on two separate data sets graded by at least 7 board-certified ophthalmologists, with the panel majority as the reference standard. For referable diabetic retinopathy the [abstract](https://pubmed.ncbi.nlm.nih.gov/27898976/) reports an area under the curve of 0.991 on EyePACS-1 and 0.990 on Messidor-2. At an operating point chosen for high specificity, sensitivity on EyePACS-1 was 90.3% with 98.1% specificity. The authors conclude that further research is needed to establish feasibility in clinical settings and whether it improves care.

**Skin cancer (2017).** The [Nature paper](https://www.nature.com/articles/nature21056) trains one convolutional network end to end, from pixels and disease labels only, on 129,450 clinical images covering 2,032 diseases, a dataset it describes as two orders of magnitude larger than earlier ones. It tests the network against 21 board-certified dermatologists on biopsy-proven images in two tasks: keratinocyte carcinomas against benign seborrheic keratoses, and malignant melanomas against benign nevi. The network performs on par with all tested experts in both. The abstract adds that mobile devices could potentially extend the reach of dermatologists beyond the clinic.

**Chest X-rays (2017).** [CheXNet](https://arxiv.org/abs/1711.05225) is a 121-layer convolutional network trained on ChestX-ray14, which contains over 100,000 frontal-view images with 14 diseases. Four practising academic radiologists annotated a test set, and the abstract reports that the network exceeds average radiologist performance on the F1 metric for pneumonia, and reaches state-of-the-art results on all 14 diseases.

## Biology and chemistry

**Protein structure (2021).** The [AlphaFold paper in Nature](https://www.nature.com/articles/s41586-021-03819-2) begins with the gap: the structures of around 100,000 unique proteins have been determined by an enormous experimental effort, a small fraction of the billions of known protein sequences, and determining one structure takes months to years. Predicting the 3D structure from the amino-acid sequence alone had been an open problem for more than 50 years. The authors report the first computational method that can regularly predict structures with atomic accuracy even when no similar structure is known, with accuracy competitive with experimental structures in a majority of cases in the 14th CASP assessment, and say the design incorporates physical and biological knowledge about protein structure.

**Generating molecules (2016).** Gomez-Bombarelli and colleagues' [paper on continuous molecular representations](https://arxiv.org/abs/1610.02415) trains an encoder, a decoder and a property predictor on hundreds of thousands of existing chemical structures, so that molecules become points in a continuous space. New structures come from decoding random vectors, perturbing known molecules or interpolating between them, and gradient-based optimisation can guide the search toward compounds with desired properties. They demonstrate it on drug-like molecules and on molecules with fewer than nine heavy atoms.

**Antibiotic discovery (2020).** Stokes and colleagues' [Cell paper](https://pubmed.ncbi.nlm.nih.gov/32084340/) trains a deep network to predict molecules with antibacterial activity, and applies it to several chemical libraries. It identified halicin, a molecule structurally divergent from conventional antibiotics, with bactericidal activity against a wide range of pathogens, and effective in mouse models of Clostridioides difficile and pan-resistant Acinetobacter baumannii infection. Separately, from 23 empirically tested predictions among more than 107 million molecules in the ZINC15 database, the model identified eight antibacterial compounds structurally distant from known antibiotics.

## Language models in medicine

The [Med-PaLM paper in Nature](https://www.nature.com/articles/s41586-023-06291-2) (2023) opens with the statement that the bar for clinical applications is high, and that automated evaluations on limited benchmarks do not meet it. It introduces MultiMedQA, a benchmark that combines six existing medical question answering data sets with a new one of questions searched online, and a human evaluation framework covering factuality, comprehension, reasoning, possible harm and bias. Flan-PaLM reaches 67.6% on MedQA, which uses US Medical Licensing Exam-style questions, more than 17 points above the prior state of the art. The authors write that human evaluation nonetheless reveals key gaps, that their instruction-prompt-tuned Med-PaLM performs encouragingly but remains inferior to clinicians, and that the results reinforce the importance of evaluation frameworks in creating safe systems.

## What the sources show

These points are the author's reading of the sources above, not claims the sources make.

- **The reference standard is part of the result.** A panel majority, 21 dermatologists, four radiologists, CASP structures and mouse infections are different kinds of evidence, and "on par with experts" means something different under each.
- **Wording becomes more careful as the output gets closer to patients.** The retinopathy abstract ends by asking whether care would improve, and the Med-PaLM abstract says it remains inferior to clinicians. Neither claims a clinical benefit.
- **Biology and chemistry papers lead with the cost of the alternative.** AlphaFold starts from months to years per structure, and the antibiotic paper starts from resistant bacteria and tests its predictions in the laboratory.
- **The data are expert labels.** The retinal study needed 54 graders, and the skin and X-ray studies relied on dermatologists and radiologists. That cost sits behind each result.

## Limits of this summary

- It covers eight papers. Genomics, pathology slides, electronic health records, surgical robots, clinical trials of deployed systems, regulation, and fairness across patient groups are left out. Whether any of these systems improves outcomes is not answered here.
- The abstracts were read, and for the AlphaFold and Med-PaLM papers also the opening paragraphs. The figures quoted are those the sources state, on their own data sets.
- Nothing in this post is medical advice.
