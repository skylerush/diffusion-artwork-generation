# Related Work

## 1. Introduction and Problem Framing

Generative image modelling asks a deceptively simple question: given a collection of example images, can a model learn the underlying data distribution well enough to synthesise new, plausible samples that were never seen during training? Over the past decade this question has driven a succession of modelling paradigms, from adversarial networks to autoregressive transformers and, most recently, to diffusion models. Our project sits squarely in this lineage, with the specific aim of generating Impressionist artwork: paintings in the manner of Monet, Renoir, Pissarro, and their contemporaries.

Learning an artistic *style*, as opposed to merely reproducing recognisable objects, is a particularly demanding instance of generative modelling. Style is diffuse and global. It lives in the statistics of brushstrokes, the broken application of complementary colours, the soft treatment of edges, and the characteristic handling of light that defines Impressionism, rather than in any single localisable object. A model must capture these high-order correlations while still producing coherent scenes. Compounding the difficulty, curated style datasets such as WikiArt Impressionism are small by modern standards (thousands, not millions, of images), so training a high-fidelity model from scratch is impractical and some form of transfer from a large pretrained prior becomes essential. Finally, style is hard to evaluate: there is no ground-truth target image for a given prompt, so quality must be assessed through distributional metrics and perceptual judgement rather than reconstruction error.

This review traces the technical foundations our project builds on. We begin with the GAN era that diffusion is now displacing, develop the theory of diffusion models from first principles, survey the engineering that made them fast and controllable, and arrive at latent diffusion and the personalization toolbox that our Phase 2 experiments depend on. We close with evaluation methodology, ethical concerns, and our intended contribution.

## 2. The GAN Era and Its Limitations

Modern generative image modelling was catalysed by Generative Adversarial Networks (Goodfellow et al., 2014), which frame generation as a minimax game between a generator that mimics the data distribution and a discriminator that learns to distinguish real from synthetic samples, both trained end-to-end by backpropagation without Markov chains. GANs produced sharp, realistic images far faster than the likelihood-based methods of their day, and a wave of engineering improvements followed. The Improved Techniques for Training GANs work (Salimans et al., 2016) introduced architectural and training tricks to stabilise the notoriously brittle adversarial objective, and along the way contributed evaluation tools still used today.

For style and art specifically, three lines of work are most relevant. Neural Style Transfer (Gatys et al., 2015) showed that the internal feature statistics of a pretrained convolutional network can separate the *content* of one image from the *style* of another and recombine them, giving the first compelling demonstration that artistic style is something a neural representation can isolate and transfer. CycleGAN (Zhu et al., 2017) extended this to unpaired image-to-image translation, learning a photo-to-painting mapping without aligned training pairs by enforcing a cycle-consistency constraint, and became a popular route to "Monet-ify" photographs. StyleGAN (Karras et al., 2018) represented the apex of GAN-based synthesis, introducing a style-based generator that disentangles high-level attributes from stochastic detail and grants scale-specific, intuitive control over the generated image.

Despite these successes, GANs carry well-known limitations. Adversarial training is unstable and prone to mode collapse, in which the generator captures only a fraction of the data distribution; convergence guarantees are weak; and conditioning the generator on rich signals such as free-form text is awkward. These shortcomings motivated the search for a generative paradigm with a more stable, likelihood-grounded training objective and better mode coverage. Diffusion models would supply exactly that, and our project deliberately chooses them over adversarial training for Impressionist art generation.

## 3. Diffusion Fundamentals

The conceptual seed of diffusion models was planted by Sohl-Dickstein et al. (2015), who drew on non-equilibrium statistical physics. Their idea is intuitive: define a *forward* process that gradually and systematically destroys the structure in data by adding a little noise at each of many steps, until the original image becomes indistinguishable from pure Gaussian noise. Because each forward step is a small, known perturbation, the process is analytically tractable. A neural network is then trained to learn the *reverse* process, removing noise one step at a time so that, starting from random noise, it can walk backwards to a clean sample drawn from the data distribution. The result is a generative model that is flexible yet tractable for learning, sampling, and likelihood evaluation.

A complementary perspective arrived with score-based modelling (Song and Ermon, 2019). Rather than reasoning about a denoising chain directly, they perturb the data with several levels of Gaussian noise and train a network to estimate the *score*, the gradient of the log-density, of each noise-perturbed distribution. New samples are then produced by annealed Langevin dynamics, which follows these score estimates from high noise toward low noise until it reaches the data manifold. This noise-conditioned, score-matching view explains *why* predicting noise gradients enables high-quality synthesis and underpins the denoising objective that diffusion models optimise.

These threads were unified and made practical by Denoising Diffusion Probabilistic Models (Ho et al., 2020). Ho and colleagues showed that a diffusion model can be trained on a simple weighted variational bound, made tractable via a connection between diffusion and denoising score matching, that reduces in practice to having the network predict the noise added at each step. With a suitable noise schedule and a U-Net backbone, DDPMs achieved state-of-the-art FID on CIFAR-10 and produced high-quality images without any adversarial loss. DDPM is the canonical formulation our Phase 1 implements from scratch, reproducing its exact training objective and noise schedule to build mechanistic understanding of the noise-to-denoise mechanism.

A continuous-time generalisation soon followed (Song et al., 2020, SDE). They recast the forward noising as a stochastic differential equation that smoothly transforms data into a known prior, with a corresponding reverse-time SDE, depending only on the score, that turns noise back into data. This framework subsumes both score-based models and DDPMs as discretisations of the same underlying process and enables new samplers, including predictor-corrector schemes and an equivalent deterministic neural ODE. It provides the conceptual map that connects the discrete DDPM we build to the broader family of denoising generators.

## 4. Faster Sampling and Guidance

Vanilla DDPMs are powerful but slow, requiring hundreds or thousands of sequential network evaluations to produce a single image, and two strands of follow-up work addressed this and the related problem of controllable generation.

On the sampling-speed front, Improved DDPM (Nichol and Dhariwal, 2021) showed that a few targeted modifications, most notably learning the reverse-process variances rather than fixing them and adopting a better (cosine) noise schedule, yield competitive log-likelihoods, improved sample quality, and roughly an order-of-magnitude reduction in the number of sampling steps. Denoising Diffusion Implicit Models (Song et al., 2020, DDIM) took a different route, defining a family of non-Markovian processes that share the DDPM training objective but admit a *deterministic* reverse sampler. DDIM produces samples 10x to 50x faster, trades compute for quality smoothly, and supports meaningful interpolation in the latent noise space. It is the accelerated sampler we use to generate Impressionist images quickly from our trained models in both phases.

On the control front, Dhariwal and Nichol (2021) demonstrated, through careful architecture ablations and a new *classifier guidance* technique, that diffusion models can beat state-of-the-art GANs on image synthesis. Classifier guidance steers sampling using gradients from a separately trained image classifier, trading diversity for fidelity. Classifier-Free Guidance (Ho and Salimans, 2022) removed the need for an auxiliary classifier altogether: by jointly training a single network in conditional and unconditional modes and linearly combining their score estimates at sampling time, one obtains the same fidelity-versus-diversity trade-off with a single tunable guidance scale. This guidance scale is precisely the knob we adjust when sampling from our fine-tuned Stable Diffusion model to control how strongly outputs adhere to Impressionist text prompts.

## 5. Latent Diffusion and Text-to-Image Systems

Running diffusion directly in pixel space is expensive, since every denoising step operates on full-resolution images. Latent Diffusion Models (Rombach et al., 2021) resolved this by first compressing images into the lower-dimensional latent space of a pretrained autoencoder, which discards imperceptible high-frequency detail while keeping the essential structure, and then performing the entire diffusion process in that compact latent space. This dramatically cuts training and inference cost while preserving quality. Crucially, the authors added cross-attention layers so the U-Net can be conditioned on external inputs such as text, enabling flexible high-resolution generation. This architecture, released publicly as Stable Diffusion, is the exact base (SD 1.5) our Phase 2 fine-tunes on WikiArt Impressionism. Its successor, SDXL (Podell et al., 2023), scales the U-Net roughly threefold, adds a second text encoder and new conditioning schemes, and appends a refinement model; while we work with SD 1.5 for tractability, SDXL illustrates how scaling and conditioning design continue to improve latent diffusion quality.

In parallel, several systems established text as the dominant conditioning signal for image diffusion. GLIDE (Nichol et al., 2021) applied diffusion to text-conditional synthesis and found classifier-free guidance produced more photorealistic, caption-aligned samples than CLIP guidance, with human raters preferring it over DALL-E. DALL-E 2, or unCLIP (Ramesh et al., 2022), introduced a two-stage pipeline in which a prior maps a caption to a CLIP image embedding and a diffusion decoder renders the image from that embedding, improving diversity while retaining photorealism and enabling image variations and language-guided edits. Imagen (Saharia et al., 2022) showed that pairing a frozen large *text-only* language model with a cascade of diffusion models, and scaling the language model in particular, drives fidelity and text-image alignment more than scaling the image model alone. Together these systems define the text-conditioning landscape that our Phase 2 prompts operate within and clarify why strong text encoders matter for style-faithful generation.

## 6. Personalization and Parameter-Efficient Fine-Tuning

Adapting a large pretrained text-to-image model to a specific style or subject, on limited data and limited GPU resources, is the central engineering challenge of our Phase 2, and a rich toolbox now exists for it.

DreamBooth (Ruiz et al., 2022) fine-tunes the full diffusion model on just a few reference images, binding a unique identifier token to a subject and adding a class-specific prior-preservation loss so the subject can be re-rendered in novel contexts without catastrophic forgetting. Textual Inversion (Gal et al., 2022) takes a lighter-weight stance: it freezes the model entirely and learns a single new pseudo-word in the text-embedding space from only three to five examples, capturing a concept or style that can then be composed into ordinary prompts. LoRA (Hu et al., 2021), originally proposed for large language models, freezes the pretrained weights and injects small trainable low-rank matrices into each layer, slashing the number of trainable parameters and the GPU memory required while matching or exceeding full fine-tuning and adding no inference latency; it has become the workhorse of cheap diffusion adaptation. Finally, ControlNet (Zhang et al., 2023) attaches a trainable copy of the diffusion encoder, connected through zero-initialised convolutions, to add spatial conditioning, edges, depth, segmentation, or pose, without disturbing the pretrained priors; it offers an optional avenue for controlling the composition of generated Impressionist images beyond text. DreamBooth, Textual Inversion, and LoRA are precisely the three fine-tuning strategies we compare in Phase 2.

## 7. Evaluation Metrics and Ethical Concerns

Because there is no single correct output for a generative model, evaluation relies on distributional and perceptual proxies. The Frechet Inception Distance (Heusel et al., 2017), introduced alongside the two time-scale update rule for GAN training, measures how close the distribution of generated images is to that of real images in a deep feature space, and is the standard metric we use to compare our generated Impressionist images against real WikiArt samples. The Inception Score and the practice of human visual evaluation (the visual Turing test) trace back to the Improved Techniques work (Salimans et al., 2016), and continue to inform how perceptual quality is judged. For text-conditioned outputs, CLIPScore (Hessel et al., 2021) provides a reference-free measure of image-text compatibility using the pretrained CLIP model, correlating better with human judgement than reference-based caption metrics; it lets us quantify how well our outputs actually match the Impressionist style prompts.

Fine-tuning on artwork also raises ethical and legal concerns. Carlini et al. (2023) demonstrated that diffusion models such as Stable Diffusion memorise and can regenerate individual training images, extracting over a thousand examples via a generate-and-filter pipeline and showing these models can be less private than GANs. Because we fine-tune on WikiArt, this directly motivates caution: the model may reproduce specific copyrighted paintings, and our evaluation must check for memorisation rather than genuine stylistic generalisation.

A final practical concern is inference cost. Consistency Models (Song et al., 2023) map noise to data directly, enabling one-step (or few-step) sampling while still allowing multistep refinement, and can be distilled from a pretrained diffusion model. Latent Consistency Models (Luo et al., 2023) adapt this to latent diffusion, predicting the solution of an augmented probability-flow ODE to achieve two-to-four-step high-resolution sampling distilled cheaply from models like Stable Diffusion, and add a Latent Consistency Fine-tuning recipe for custom datasets. These offer a fast-generation path relevant to deploying our Impressionist generator efficiently.

## 8. Positioning and Our Intended Contribution

The literature above reveals a clear arc: from unstable adversarial generators, through the physically motivated forward-and-reverse diffusion of Sohl-Dickstein et al. (2015) and its practical realisation in DDPM (Ho et al., 2020), to the efficient, controllable, text-conditioned latent diffusion of Rombach et al. (2021) and the personalization tools that adapt it.

Our project occupies two complementary positions on this arc. **Phase 1** is pedagogical and mechanistic: we implement a DDPM from scratch, reproducing the forward noising schedule and the noise-prediction objective, to build first-hand understanding of *why* the noise-to-denoise mechanism works, before relying on any high-level library. **Phase 2** is empirical and comparative: rather than simply applying one fine-tuning recipe, we conduct a documented multi-run comparison of LoRA (Hu et al., 2021), full fine-tuning, and DreamBooth (Ruiz et al., 2022) for adapting Stable Diffusion 1.5 to Impressionism on WikiArt, evaluated with FID (Heusel et al., 2017) and CLIPScore (Hessel et al., 2021) and audited for memorisation (Carlini et al., 2023).

Our intended contribution is therefore less a new architecture than a transparent account of *process*. We deliberately record failed runs, hyper-parameter searches, and the analysis that explains them, so that the trade-offs between parameter-efficient and full adaptation on a small artistic dataset and constrained hardware are made legible. In a field where polished final results dominate publication, a careful, reproducible study of how one actually coaxes Impressionist style out of a pretrained diffusion model is the gap we aim to fill.

## References

Goodfellow et al. (2014). *Generative Adversarial Networks*. arXiv:1406.2661.

Salimans et al. (2016). *Improved Techniques for Training GANs*. arXiv:1606.03498.

Gatys et al. (2015). *A Neural Algorithm of Artistic Style*. arXiv:1508.06576.

Zhu et al. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. arXiv:1703.10593.

Karras et al. (2018). *A Style-Based Generator Architecture for Generative Adversarial Networks*. arXiv:1812.04948.

Sohl-Dickstein et al. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. arXiv:1503.03585.

Song and Ermon (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. arXiv:1907.05600.

Ho et al. (2020). *Denoising Diffusion Probabilistic Models*. arXiv:2006.11239.

Song et al. (2020). *Score-Based Generative Modeling through Stochastic Differential Equations*. arXiv:2011.13456.

Nichol and Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models*. arXiv:2102.09672.

Song et al. (2020). *Denoising Diffusion Implicit Models*. arXiv:2010.02502.

Dhariwal and Nichol (2021). *Diffusion Models Beat GANs on Image Synthesis*. arXiv:2105.05233.

Ho and Salimans (2022). *Classifier-Free Diffusion Guidance*. arXiv:2207.12598.

Rombach et al. (2021). *High-Resolution Image Synthesis with Latent Diffusion Models*. arXiv:2112.10752.

Nichol et al. (2021). *GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models*. arXiv:2112.10741.

Ramesh et al. (2022). *Hierarchical Text-Conditional Image Generation with CLIP Latents*. arXiv:2204.06125.

Saharia et al. (2022). *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding*. arXiv:2205.11487.

Podell et al. (2023). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*. arXiv:2307.01952.

Ruiz et al. (2022). *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*. arXiv:2208.12242.

Gal et al. (2022). *An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion*. arXiv:2208.01618.

Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.

Zhang et al. (2023). *Adding Conditional Control to Text-to-Image Diffusion Models*. arXiv:2302.05543.

Heusel et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. arXiv:1706.08500.

Hessel et al. (2021). *CLIPScore: A Reference-free Evaluation Metric for Image Captioning*. arXiv:2104.08718.

Carlini et al. (2023). *Extracting Training Data from Diffusion Models*. arXiv:2301.13188.

Song et al. (2023). *Consistency Models*. arXiv:2303.01469.

Luo et al. (2023). *Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference*. arXiv:2310.04378.
