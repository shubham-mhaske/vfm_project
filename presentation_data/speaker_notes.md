# Speaker Notes: Promptable Pathology Segmentation
## Graduate Course Final Project Presentation
## CSCE 689 Fall 2025 | Shubham Mhaske

---

## SLIDE 1: Title Slide
### "Promptable Pathology: Zero-Shot Medical Image Segmentation"

**[Opening - 30 seconds]**

> "Good morning/afternoon. Today I will be presenting my final project: Promptable Pathology—an interactive zero-shot pipeline for medical image analysis.

> This project addresses a fundamental question in applied deep learning: **When working with limited labeled data, is it more effective to finetune pretrained models or to engineer better prompts?**

> I'll show you that for small medical datasets, prompt engineering consistently outperforms model finetuning—a finding that has significant implications for deploying AI in clinical settings."

**Slide Content:**
- Title: "Promptable Pathology"
- Subtitle: "Zero-Shot Medical Image Segmentation"
- Course: CSCE 689 Fall 2025 | Shubham Mhaske

**Key points to emphasize:**
- Frame as a research question, not just a demo
- "Interactive" means user-in-the-loop with prompts
- Set up the finetuning vs. prompting comparison early

---

## SLIDE 2: Methodology
### "Two-Stage Pipeline: SAM2 + CLIP"
### Figure: fig4_method_overview.png

**[2-3 minutes]**

> "The proposed approach uses a modular two-stage pipeline that decouples segmentation from classification. Let me walk you through this figure.

> **[Point to left side of figure]** On the input side, we have histopathology images from the BCSS dataset—these are H&E stained breast cancer tissue sections. The user—imagine a pathologist—provides a prompt: either clicking points of interest or drawing a bounding box around a region.

> **[Point to SAM2 block]** **Stage 1: Segmentation using SAM2.** The Segment Anything Model version 2 takes the image and prompt, then generates a binary segmentation mask. SAM2 uses a Hiera-Large backbone with 224 million parameters. It was pretrained on the SA-1B dataset—11 million images with 1.1 billion masks. This massive pretraining gives it strong generalization to unseen domains like pathology.

> **[Point to CLIP block]** **Stage 2: Classification using CLIP.** The segmented region is cropped out and passed to CLIP—Contrastive Language-Image Pretraining. CLIP compares the image embedding against text prompt embeddings for each tissue class. It was trained on 400 million image-text pairs from the internet.

> **[Point to output]** We classify into five tissue types: tumor epithelium, stroma, lymphocytic infiltrate, necrosis, and blood vessels.

> **About the dataset:** The BCSS dataset contains 151 image patches from The Cancer Genome Atlas, split into 85 training, 21 validation, and 45 test images. The test set uses held-out patient cases to ensure generalization.

> **Critical challenge:** The class distribution is highly imbalanced—stroma comprises 54% of pixels while lymphocytes are only 3%. This imbalance affects both training and evaluation."

**Slide Bullet Points:**
- "Modular Two-Stage Design: SAM2 (Segmentation) + CLIP (Classification)"
- "User provides prompts (points/boxes) → System segments and classifies tissue regions"
- "Dataset: BCSS (151 images, 5 tissue classes: tumor, stroma, lymphocyte, necrosis, blood vessel)"

**Technical details to know if asked:**
- SAM2 Hiera-L: 224M parameters, trained on SA-1B (11M images, 1.1B masks)
- CLIP ViT-B/32: 151M parameters, trained on WebImageText (400M pairs)
- BCSS: 5 classes, 151 images at 0.25 μm/pixel resolution
- Train/Val/Test: 85/21/45 with patient-level separation
- Class distribution: Stroma 54%, Tumor 25%, Necrosis 12%, Blood 6%, Lymphocytes 3%

---

## SLIDE 3: The Finetuning Hypothesis
### "Why Finetuning Failed"
### Figure: fig3_training_analysis.png

**[3-4 minutes]**

> "Before presenting what worked, I want to discuss what didn't work—because these negative results are equally informative and represent most of the project effort.

> **[Point to figure showing training curves]** This figure shows training curves from our finetuning experiments. Notice how training loss decreases but validation loss stays flat or increases—classic overfitting.

> **Our initial hypothesis was intuitive:** Medical images have domain-specific characteristics—unique textures, staining patterns, cellular structures. Surely, finetuning these massive foundation models on pathology data should improve performance, right?

> **We tested this hypothesis with five different approaches:**

> **[Point to first data point]** **1. PathSAM2 with CTransPath Embeddings:** CTransPath is a self-supervised vision transformer pretrained on 15 million histopathology patches. We integrated its embeddings into SAM2's prompt encoder to provide domain-specific visual features. Result: 0.37 Dice—that's 33% worse than zero-shot SAM2.

> **2. LoRA Finetuning (rank=8):** Low-Rank Adaptation adds small trainable matrices to attention layers while freezing the base model. We added approximately 4 million parameters and trained for 50 epochs with careful learning rate scheduling. Result: 0.27 Dice—a 51% degradation. The model forgot how to segment entirely.

> **3. LoRA-Light (rank=4):** We hypothesized rank=8 was overparameterized. Reducing to rank=4 yielded 0.36 Dice—still 35% worse than zero-shot.

> **4. Focal Loss Training:** Focal loss down-weights easy examples to handle class imbalance. Combined with box prompts: 0.37 Dice—no improvement over PathSAM2.

> **[Emphasize this point]** **5. Linear Probe on CLIP Features:** We also ran a supervised classification baseline—training a Logistic Regression classifier directly on CLIP embeddings extracted from ground truth regions. This achieved 40.4% accuracy—which is still worse than our zero-shot LLM-prompted approach at 44.4%.

> **This last finding is crucial:** Even when we train a supervised classifier on the exact same features, natural language prompts from an LLM provide better supervision than gradient descent on small data.

> **Why did all finetuning fail?** The answer is **catastrophic forgetting** due to data ratio mismatch. SAM2 was pretrained on 11 million images; our training set has 85 images. That's a ratio of 130,000:1. The model doesn't have enough examples to learn new patterns—it just forgets what it knew.

> **Key finding:** Zero-shot SAM2 achieves 0.55 Dice, outperforming ALL finetuned variants by 30-50%."

**Slide Bullet Points:**
- "Tested: PathSAM2+CTransPath (0.37), LoRA adapters (0.27-0.36), Focal Loss (0.37)"
- "Linear Probe: Logistic Regression on CLIP features (40.4%) — still < Zero-Shot (44.4%)"
- "Zero-Shot SAM2 (0.55) outperforms ALL finetuned models by 30-50%"

**If asked about preventing overfitting:**
- We tried learning rate scheduling (cosine annealing, warmup)
- Early stopping with patience=10
- Weight decay regularization (0.01, 0.001)
- LoRA was specifically chosen because it freezes 99% of parameters
- The fundamental issue is data quantity, not hyperparameters
- Even supervised Linear Probe on CLIP features (40.4%) was beaten by zero-shot LLM prompts (44.4%)

---

## SLIDE 4: Segmentation Results
### "Prompt Strategy Comparison"
### Figure: fig1_segmentation_comprehensive.png

**[3-4 minutes]**

> "Given that finetuning was counterproductive, we pivoted to optimizing the prompting strategy for zero-shot SAM2. This figure shows our systematic evaluation.

> **[Point to bar chart in figure]** We evaluated four prompting strategies, and the performance gap is dramatic:

> **[Point to first bar]** **1. Centroid Point Prompt:** Place a single point at each region's geometric center. This achieves 0.34 Dice—our baseline. The limitation is ambiguity: a single point provides no information about region extent or shape. SAM2 has to guess how big the region should be.

> **[Point to second bar]** **2. Multi-Point Sampling:** Sample 5 random points within each region. This improves to 0.42 Dice—a 24% relative improvement. More points give SAM2 better coverage, but we're still only saying 'include these pixels'—we're not saying 'exclude those pixels.'

> **[Point to third bar]** **3. Bounding Box Prompt:** Provide a tight bounding box around each region. This achieves 0.55 Dice—64% improvement over centroid, 31% over multi-point. **This is the key insight:** Bounding boxes implicitly define negative space. Everything outside the box is definitely background. SAM2 now has both inclusion AND exclusion information.

> **[Point to fourth bar]** **4. Box with Negative Points:** Add explicit negative points outside the region—placed at corners of an expanded box. This achieves 0.555 Dice. Marginal but consistent improvement.

> **[Emphasize this technical detail]** **TTA with Geometric Prompt Transformation:** We implemented Test-Time Augmentation using horizontal flip, vertical flip, and 90-degree rotation. But here's the engineering challenge: standard TTA breaks prompt-based models because when you flip the image, the bounding box coordinates no longer align with the tissue.

> We wrote a custom wrapper in `src/tta_utils.py` that geometrically transforms the prompt coordinates in sync with each image augmentation—rotating box corners, flipping point coordinates. This ensures 100% alignment between transformed images and transformed prompts. This engineering detail adds approximately 2 percentage points, yielding 0.566 Dice.

> **[Point to MedSAM comparison if in figure]** **Comparison with MedSAM:** MedSAM is the full ViT-B model finetuned on 1.5 million medical image masks from 11 imaging modalities—CT, MRI, X-ray, ultrasound, and histopathology. With box prompts, MedSAM achieves 0.52 Dice; with our TTA, 0.54 Dice.

> **Our zero-shot SAM2 at 0.566 outperforms the fully supervised MedSAM trained on 1.5 million masks.** This validates our thesis: for small datasets, prompt engineering beats domain finetuning."

**Slide Bullet Points:**
- "Box prompts: 0.55 Dice vs Centroid: 0.34 (+64% improvement)"
- "TTA with Prompt Rotation: Geometrically transforming box coords to match augmentations (+2%)"
- "Ours (0.555) beats fully supervised MedSAM ViT-B Model (0.536)"

**Technical details:**
- Dice coefficient: 2|A∩B|/(|A|+|B|), range [0,1], 1.0 is perfect overlap
- TTA: geometric mean of 4 predictions (original + 3 augmentations) with synchronized prompt coordinate transformation
- Negative points: 4 points at box corners offset by 10 pixels outward
- MedSAM: Full ViT-B encoder (93.7M params), trained on 1.5M masks from 11 medical imaging modalities
- Our SAM2: Hiera-L (224M params), zero-shot, no medical training

---

## SLIDE 5: Qualitative Results
### "Visual Comparison Across Methods"
### Figure: best_method_comparison.png

**[2-3 minutes]**

> "The quantitative improvements translate to dramatic visual differences. This figure shows our best-performing samples—cases specifically selected where the prompt strategy made the biggest impact.

> **[Point to image columns]** Each row is a different tissue sample. From left to right: original H&E image, ground truth annotation, then predictions from each prompt strategy.

> **[Point to centroid column]** **Centroid prompts** produce what I call 'blobs'—diffuse, roughly circular regions. Look at this tumor sample: the centroid prompt captures a vague circular area, but the actual tumor has irregular, infiltrating boundaries. The model defaults to simple shapes because a single point provides no shape information.

> **[Point to multi-point column]** **Multi-point prompts** improve coverage—you can see the predictions expand to cover more of the tissue—but boundaries remain imprecise. The edges are smooth approximations rather than following actual tissue contours.

> **[Point to box column]** **Box prompts** show dramatically better boundary adherence. Look at this same tumor: now the segmentation follows the actual tumor margin, capturing the irregular infiltrating edge. The box told SAM2 exactly where to look, and it found the precise boundary within that region.

> **[Point to box+neg column]** **Box with negative points** provides incremental refinement. Look at tissue interfaces where tumor meets stroma. The boundaries are cleaner, with less 'bleeding' into adjacent tissue. The negative points explicitly tell SAM2 'this is definitely NOT the region of interest.'

> **[Emphasize clinical relevance]** Why does this matter clinically? In quantitative pathology, we measure tumor burden, analyze margins, compute morphometric features. A blob that roughly covers the tumor gives you wrong area measurements, wrong perimeter estimates, wrong shape descriptors. You need precise boundary delineation for any downstream analysis."

**Slide Bullet Points:**
- "Best samples: Ours (Box+Neg) achieves tight boundaries vs blob-like point prompts"
- "Centroid prompts fail on multi-region classes; Box captures full extent"
- "Negative points suppress background bleeding in complex tissue interfaces"

**Key observations to point out in figure:**
- Point prompts → circular/blob predictions (show specific example)
- Box prompts → boundary-aware predictions following tissue contours
- Negative points → cleaner interfaces between tissue types
- Color overlay helps visualize prediction vs ground truth alignment

---

## SLIDE 6: Classification Results
### "CLIP Prompt Engineering"
### Figure: fig2_clip_analysis.png

**[3-4 minutes]**

> "Now let's talk about the classification stage. Once we have a good segmentation, we need to identify the tissue type. This is where CLIP comes in—and where prompt engineering becomes even more critical.

> **[Point to figure showing prompt evolution/accuracy bars]** This figure shows our progression through three prompt versions and the dramatic accuracy improvements.

> **The fundamental challenge:** CLIP was trained on internet images with natural language captions—photos of cats, food, landscapes, with captions like 'a fluffy orange cat sitting on a couch.' It has never seen a histopathology image. It has never read a pathology textbook. How do we teach it to recognize tumor from stroma?

> **[Point to V1 bar]** **Version 1 - Medical Terminology:** Our first attempt used proper pathology terminology. We wrote prompts like:
> - 'Neoplastic tumor epithelium with high nuclear-to-cytoplasmic ratio'
> - 'Desmoplastic stromal reaction with collagenous extracellular matrix'
> - 'Lymphocytic infiltrate with small round blue cells'
> 
> These are accurate descriptions—any pathologist would recognize them. Result: **12.2% accuracy**. That's barely above random chance for 5 classes (20% would be random).

> Why did this fail? CLIP's embedding space has no grounding for words like 'desmoplastic,' 'neoplastic,' or 'nuclear-to-cytoplasmic ratio.' These terms never appear in internet captions. We're speaking a language CLIP doesn't understand.

> **[Point to V2 bar]** **Version 2 - Visual Descriptors:** We used Google's Gemini to translate medical terminology into visual descriptions that might appear in internet text:
> - 'Dense clusters of dark purple cells packed tightly together'
> - 'Pink wavy fibers running in parallel bands'
> - 'Scattered small round blue dots'
> 
> Result: **32% accuracy**—a 162% relative improvement. Now CLIP understands colors, textures, patterns.

> **[Point to V3 bar]** **Version 3 - Few-Shot Grounding:** We provided Gemini with actual example images from our BCSS dataset along with their labels, and asked it to describe what visual features distinguish each class in these specific images. This grounds the prompts in our actual data distribution, not generic pathology descriptions.
> 
> Result: **44.4% accuracy**—that's **264% improvement** over Version 1.

> **[Mention the alternatives tested]** We also tested:
> - **PLIP:** A CLIP model finetuned specifically on pathology image-text pairs from PubMed. Achieved only 27%—the training distribution didn't match our specific tissue classification task.
> - **CLIP+PLIP Ensemble:** Averaging predictions yielded 31%. PLIP's confident wrong answers dominated the ensemble.

> **The key insight parallels segmentation:** Domain-specific models don't automatically win. Aligning prompts with the model's learned representation space is more valuable than domain finetuning on mismatched data."

**Slide Bullet Points:**
- "V1 'Medical Jargon' (12%) → V3 'Gemini Few-Shot' (44%) = +264% improvement"
- "Key: Translate pathology terms to visual descriptors CLIP understands"
- "Also tested: PLIP (27%), Ensembles (31%) - CLIP alone is best"

**Prompt examples to mention if asked:**
- V1 Tumor: "Neoplastic tumor epithelium with high nuclear-to-cytoplasmic ratio"
- V3 Tumor: "Dense irregular clusters of dark purple cells with visible nuclei"
- V1 Stroma: "Desmoplastic stromal reaction with collagenous tissue"
- V3 Stroma: "Pink wavy fibers and elongated cells in parallel arrangement"

**Why PLIP failed:**
- Trained on PubMed pathology papers—different vocabulary than our visual descriptors
- Images were figure panels, not raw WSI patches
- Classification task (5 tissue types) differs from PLIP's training objective

---

## SLIDE 7: Per-Class Analysis
### "Class Imbalance & Performance"
### Figure: best_per_class.png

**[2-3 minutes]**

> "Aggregate metrics can be misleading. Let's break down performance by tissue class—this reveals important patterns and limitations.

> **[Point to figure showing per-class results]** This figure shows our best-performing examples for each tissue type, along with Dice scores.

> **[Point to high-performing classes]** **High-performing classes:**
> - **Tumor:** 0.70 Dice—our best class. Tumor regions are typically compact, cohesive masses with distinctive dark purple staining from densely packed nuclei. The irregular but connected shape is well-suited to box prompts.
> - **Stroma:** 0.60 Dice. Stroma is the majority class at 54% of pixels—so the model sees abundant examples. Its pink, wavy, fibrous texture is visually distinctive from other classes.
> - **Necrosis:** 0.55 Dice. Dead tissue has a characteristic pale, structureless, washed-out appearance—easy to distinguish visually.

> **[Point to challenging classes]** **Challenging classes:**
> - **Blood vessels:** 0.42 Dice. Vessels are thin, elongated, tubular structures. A bounding box around a vessel includes substantial surrounding stroma—poor signal-to-noise ratio for the prompt.
> - **Lymphocytes:** 0.25 Dice—our worst class. Lymphocytes are tiny cells, often appearing as scattered individual dots or small clusters. They represent only 3% of pixels—severe class imbalance means few training examples.

> **[Emphasize the clinical paradox]** Here's the clinical paradox: **lymphocytes are often the most important class.** They indicate immune response. Tumor-infiltrating lymphocytes (TILs) are used for immunoscoring in breast cancer prognosis. PD-L1 status depends on lymphocyte presence. Our pipeline struggles with exactly the class that matters most for immunotherapy decisions.

> **Why does class imbalance hurt so much?** With 54% stroma and 3% lymphocytes, our aggregate Dice is dominated by stroma performance. A model that perfectly segments stroma but completely ignores lymphocytes would still achieve a reasonable-looking average score.

> **Future directions to address this:**
> - Class-adaptive prompting: Use point prompts for lymphocytes (they're small), box prompts for tumor (they're large)
> - Stratified evaluation: Report per-class metrics prominently, not just aggregate
> - Targeted augmentation: Oversample rare class examples during any finetuning"

**Slide Bullet Points:**
- "Best: Tumor (0.70 Dice), Stroma (0.60) | Worst: Lymphocytes (0.25), Blood (0.42)"
- "Class imbalance (Stroma=54%, Lymphocytes=3%) drives performance gap"
- "Rare classes need domain-specific prompts or targeted augmentation"

**Class frequencies for reference:**
- Stroma: 54% of pixels
- Tumor: 25%
- Necrosis: 12%
- Blood vessels: 6%
- Lymphocytes: 3%

**Clinical importance of lymphocytes:**
- Tumor-infiltrating lymphocytes (TILs) correlate with prognosis
- Immunoscoring uses lymphocyte density
- PD-L1 checkpoint inhibitor decisions involve lymphocyte analysis

---

## SLIDE 8: Conclusions
### "Key Findings & Takeaways"
### Figure: fig5_summary_results.png (if present)

**[2-3 minutes]**

> "Let me synthesize the main findings from this project.

> **[Point to Key Finding checkmark]** **Finding 1: For small medical datasets, Prompt Engineering outperforms Model Finetuning.**

> This challenges conventional wisdom. We're trained to think 'domain-specific finetuning = better performance.' But with N<100 training images, finetuning large foundation models leads to catastrophic forgetting. SAM2 was pretrained on 11 million images—our 85 training images are noise in comparison. The model forgets its general segmentation capabilities faster than it learns pathology-specific patterns.

> **The right strategy:** Preserve the pretrained knowledge. Find better ways to communicate with the model through prompts.

> **[Point to Segmentation checkmark]** **Finding 2: Zero-shot SAM2 with Box+Negative prompts achieves 0.555 Dice.**

> This beats every finetuned approach we tried. It beats PathSAM2 with pathology-specific embeddings. It beats LoRA adapters. And critically, it beats MedSAM—a model specifically finetuned on 1.5 million medical images. Prompt strategy matters more than domain-specific training.

> **[Point to Classification checkmark]** **Finding 3: Zero-shot CLIP with LLM prompts beats supervised classifiers.**

> We improved from 12.2% to 44.4% accuracy by translating medical jargon to visual descriptors. That's 264% improvement just from better prompts.

> But the more surprising result: our zero-shot LLM-prompted approach at 44.4% **beats a Linear Probe trained directly on CLIP features** at 40.4%. Natural language prompts from Gemini provide better supervision than gradient descent on labeled data. That's a profound statement about LLMs as supervision signals.

> **[Point to Insight checkmark]** **Finding 4: 50+ failed experiments informed these conclusions.**

> Research is mostly failure. PathSAM2, LoRA variants, focal loss, PLIP, ensembles—most approaches didn't work. But each failure narrowed the hypothesis space and pointed toward prompt engineering as the winning strategy.

> **[If time permits, broader implications]** **Broader implications for medical AI:**
> - Foundation models contain substantial transferable knowledge—even to domains far from their training
> - Prompt engineering is a form of model adaptation that doesn't risk forgetting
> - For clinical deployment, this means faster iteration without massive labeled datasets

> Thank you for your attention. I'm happy to answer questions about any aspect of this work—the methodology, specific experiments, or implications for medical imaging."

**Slide Bullet Points (checkmarks):**
- ✓ Key Finding: "For small medical datasets (N<100), Prompt Engineering > Model Finetuning"
- ✓ Segmentation: "Zero-Shot SAM2 + Box+Neg: 0.555 Dice (beats finetuned MedSAM 0.536)"
- ✓ Classification: "CLIP + Gemini Few-Shot Prompts: 44.4% Accuracy (+264% vs jargon)"
- ✓ Insight: "50+ failed experiments → Simple zero-shot with good prompts wins"

**Limitations to acknowledge if asked:**
- Per-class performance varies 3x (0.25 to 0.70)—minority classes need work
- Classification at 44% still has room for improvement
- Single dataset (BCSS)—needs validation on other pathology datasets
- Interactive system requires user prompts—not fully automated

**Future work directions:**
- Class-adaptive prompting (point prompts for lymphocytes)
- Multi-round interactive refinement
- Extension to other cancer types and imaging modalities
- Prompt optimization via gradient-based methods (soft prompts)

---

## Q&A PREPARATION

### Anticipated Questions and Detailed Answers:

**Q: Why not collect more training data?**
> "Medical image annotation requires expert pathologists—typically costing $50-100 per hour—and each image takes 10-30 minutes to annotate densely. BCSS with 151 images represents significant annotation effort. More importantly, our findings are specifically relevant to the common scenario in medical imaging where labeled data is limited. Many rare diseases, new imaging modalities, or institution-specific protocols face this same constraint. The question isn't 'can we get more data?' but 'what works when we can't?'"

**Q: How does this compare to fully supervised methods like U-Net?**
> "Fully supervised U-Net models trained on BCSS with extensive augmentation report approximately 0.60-0.65 Dice. Our zero-shot approach at 0.555-0.566 is competitive, and we achieve this without any training on the target dataset. The key advantage is rapid deployment: if a hospital has a new tissue type or staining protocol, they can use our system immediately by adjusting prompts, rather than collecting and annotating thousands of images for retraining."

**Q: Would larger SAM models help?**
> "We used SAM2 Hiera-Large—224 million parameters—which is the largest public variant. SAM2 Hiera-Base and Hiera-Small show similar patterns: finetuning hurts, prompting helps. The issue isn't model capacity but data-to-parameter ratio. With 85 training images, even a 10M parameter model would have 100,000x more parameters than data points."

**Q: Why use CLIP instead of training a pathology-specific classifier?**
> "We tried exactly that with the Linear Probe baseline—Logistic Regression on CLIP features using ground truth labels. It achieved 40.4%, below our zero-shot 44.4%. We also tested PLIP, a CLIP model finetuned on PubMed pathology figures, and it only reached 27%. The modular two-stage design also allows independent improvement—we can upgrade SAM or CLIP separately as better models release."

**Q: What are the computational requirements?**
> "Inference runs in approximately 1-2 seconds per image on an A100 GPU. SAM2's image encoder is the bottleneck at ~800ms; the prompt encoder and mask decoder are <100ms; CLIP classification is ~50ms. The pipeline is practical for interactive clinical use. Training our failed finetuning experiments took 4-8 hours per run on a single A100."

**Q: Could you combine prompting with limited finetuning?**
> "This is exactly the direction I'd explore next. Approaches like prompt tuning—learning continuous prompt vectors while freezing the model—or adapter layers inserted between frozen blocks could potentially get benefits of both. The key is keeping the base model frozen to prevent forgetting. Recent work on 'prompt learning' in vision-language models shows promise here."

**Q: How did you implement the geometric TTA for prompts?**
> "Standard TTA fails because if you horizontally flip an image, a box at [100, 200, 300, 400] is now pointing to the wrong region. We implemented coordinate transformation functions that, for each augmentation type, compute the corresponding prompt transformation:
> - Horizontal flip: new_x = image_width - old_x
> - Vertical flip: new_y = image_height - old_y
> - 90° rotation: (x, y) → (image_height - y, x)
> The predictions from all 4 augmentations are inverse-transformed back to original coordinates and averaged. This is in `src/tta_utils.py`."

**Q: Why does the Linear Probe lose to zero-shot LLM prompts?**
> "This was our most surprising finding. I believe it's because Logistic Regression on 85 training samples is prone to overfitting—especially with 768-dimensional CLIP features. The LLM-generated prompts, in contrast, encode visual semantics learned from billions of text tokens. The LLM has 'seen' descriptions of pink fibers, purple cells, scattered dots in countless contexts. It's effectively leveraging a much larger 'training set' of language knowledge to generate discriminative descriptions."

**Q: What would you do differently with more time?**
> "Three things: (1) Class-adaptive prompting—automatically select point vs box prompts based on predicted class size. (2) Interactive refinement—let users correct predictions and use those corrections as additional prompts. (3) Cross-dataset validation—test on NuCLS, PanNuke, or other pathology datasets to verify findings generalize."

---

## TIMING GUIDE

| Slide | Content | Target Time | Cumulative |
|-------|---------|-------------|------------|
| 1 | Title & Introduction | 0:30 | 0:30 |
| 2 | Methodology & Pipeline | 2:30 | 3:00 |
| 3 | Failed Experiments (The Iceberg) | 3:00 | 6:00 |
| 4 | Segmentation Results | 3:00 | 9:00 |
| 5 | Visual Evidence | 2:00 | 11:00 |
| 6 | CLIP Prompt Engineering | 3:00 | 14:00 |
| 7 | Per-Class Analysis | 2:00 | 16:00 |
| 8 | Summary & Takeaways | 2:00 | 18:00 |
| - | **Total** | **~18 minutes** | |

**Adjusting for time constraints:**

**For 15-minute slot:**
- Condense Slide 5 (Visual Evidence) to 1 minute—just show the figure and highlight one example
- Condense Slide 7 (Per-Class) to 1 minute—mention the imbalance issue briefly
- Skip detailed prompt examples in Slide 6
- Total: ~15 minutes

**For 20-minute slot:**
- Full presentation as scripted
- Leave 2 minutes for questions

**For 25-30 minute slot:**
- Expand Slide 3 with more detail on each failed experiment
- Show additional qualitative examples in Slide 5
- Discuss the TTA implementation in more technical depth
- Leave 5-7 minutes for Q&A

---

## PRESENTATION GUIDANCE

### Delivery Tips:

1. **Open with the research question:** "Should you finetune or prompt engineer?" hooks the audience immediately

2. **Use the figures actively:** Point to specific bars, curves, image regions. Don't just describe—direct attention

3. **Tell the failure story with pride:** Slide 3 is your credibility builder. "We tried everything and it failed" shows rigor. The punchline—zero-shot wins—is more powerful after the struggle

4. **Pause for key numbers:** When you say "0.555 Dice" or "264% improvement," pause. Let it register. Don't rush through the impact

5. **Connect findings across slides:** "Just like with segmentation, the domain-finetuned model didn't help" reinforces the theme

6. **Acknowledge limitations honestly:** Mentioning the lymphocyte challenge shows scientific maturity

7. **End with implications, not just results:** "This suggests a path forward for medical AI deployment" leaves the audience with something to think about

### Common Pitfalls to Avoid:

- Don't read bullet points verbatim—expand and explain
- Don't skip the failed experiments—they're the scientific meat
- Don't apologize for limitations—present them as future work
- Don't rush the summary—it's what people remember

### Technical Terms to Define (if audience is unfamiliar):

- **Dice coefficient:** Overlap metric, 2×intersection/sum of areas, 1.0 = perfect
- **Zero-shot:** No training on target task; use model as-is
- **Prompt engineering:** Designing inputs to guide model behavior
- **Catastrophic forgetting:** Finetuning destroys pretrained knowledge
- **LoRA:** Low-Rank Adaptation—efficient finetuning method
- **TTA:** Test-Time Augmentation—averaging predictions on transformed inputs

---

*Prepared for: CSCE 689 Fall 2025 Final Project Presentation*
*Presenter: Shubham Mhaske*
*Date: December 2025*
