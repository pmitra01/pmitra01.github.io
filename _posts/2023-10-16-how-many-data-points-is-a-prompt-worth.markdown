---
layout: distill
title:  "How many data points is a prompt worth?"
author: Payal
date:   2023-10-16 23:46:17 +0100

---
## TL;DR
This is a simple post that journals two things. 

First, the long urge I have had to conduct a basic experiment to see for myself 'how many (training) data points a 
(training) prompt was worth'. Given a task we wish to train an LLM for, can we yield comparable or even better 
models with a mere fraction of the labeled data points traditionally used by simply transforming the training 
examples differently? I share the experiment setup, code and observations from fine-tuning a series of models on the 
rudimentary task of topic classification which illustrates the sample efficiency of instruction tuning with prompts.

Second, it consolidates reflections on what makes prompting powerful: (1) tracing how the use of prompting an LLM 
for completion is at the heart of today's class of LLM fine-tuning methods (instruction tuning, prefix tuning, RLHF, 
etc). It makes the fine-tuning objective more consistent with the pretraining objective; (2) how non-generative 
tasks can be creatively reformulated as generative tasks (3) when to consider fine-tuning over prompt engineering

Note: I'm not making a case for 'LLM-maximalism' where the perfect prompt and best-on-market LLM combo magically 
solves compound NLP tasks. On the contrary to build reliable and feasible production-grade systems with the 
relatively untamed LLMs, I identify with: (i) [LLM-Pragmatism](https://explosion.ai/blog/against-llm-maximalism) [1] 
and [Evaluation-Driven-Development](https://eugeneyan.com/writing/llm-patterns/?utm_source=convertkit&utm_medium=email&utm_campaign=How+to+Match+LLM+Patterns+to+Problems%20-%2011495339#evals-to-measure-performance) [2] 
(EDD?). Depending on your use-case LLM Pragmatism could call for some combination of task-composibility and 
control-flows that coordinate different components and tools. <d-footnote>Instead of aggregating all steps to be 
performed by the product feature into one big LLM prompt, decompose it into subtasks and delegate different 
components to heuristic or more explainable ML solutions where best suited, and LLM-generation only for certain 
aspects such as summarisation or intent interpretation, which are highly abstractive in nature. The decision flow of 
control between different components can be determined with agents or rules as seen in immplementation patterns such as 
RAG (Retrieval Augmented Generation) in search, or agent-based LLM workflows. Several other examples can also be 
found in blog posts `Building LLM applications for production` by Chip Huyen [3] and `Patterns for Building 
LLM-Based systems and products` by Eugene Yan [1].</d-footnote>. (ii) It is important to understand when and how to 
fine-tune a suitable (possibly 'small') model for one's use case.


## Diving in
The genesis of this blog post was in early November 2022 after attending an [AKBC](https://www.akbc.ws/2022/) 
workshop. During two of the keynote talks [4] <d-footnote>Keynote talks [4] by Prof. Heng Ji introducing 
[Code4Struct](https://arxiv.org/abs/2210.12810) [5]; and Prof. Eneko Agirre on ['Pretrain, 
Prompt, Entail'](https://dl.acm.org/doi/abs/10.1145/3477495.3532786)[6] paradigm for information extraction.
</d-footnote>, it first dawned upon me how simple, versatile and effective the tactic of prompting was not just for 
in-context learning at inference time, but also to further fine-tune LLMs on desired tasks. Prompts allowed the 
framing of tasks in a way that was closer to the pretraining objective (next token/sentence prediction) for certain 
LLM architectures. All of a sudden I felt as though I had been looking at so many of the NLP tasks around knowledge 
extraction unhelpfully for years. Made me think about some problems from first principles again.

But then the ChatGPT-era happened and this newfound wonder with prompts felt antiquated, followed by personal 
genAI-fatigue. However, I am ready to revive this article because (1) I had to scratch the itch, (2) the intuition 
of prompt guided learning is still fundamental in (almost all?) approaches to fine-tuning LLMs and can often feel 
hidden beneath the abstractions of handy libraries and APIs around generative models. 

The structure of the post is as follows:
1. [Experiment on sample-efficiency of prompt-based fine-tuning Vs. conventional methods](#1-experiment)
2. [Reflections on fine-tuning LLMs with prompts](#2-reflections)
3. [Observatory Note on use of prompt engineering vs fine-tuning](#3-takeaway-remarks)


<h3 id="1-experiment">Experiment</h3>
Experiment Objective: To quantitatively observe the sample efficiency of fine-tuning LLMs using prompt-completion pairs 
instead of conventional supervised classification on input-output labelled data points, I performed an experiment 
inspired by the paper '[How many data points is a prompt worth]((https://arxiv.org/abs/2103.08493))' [7]. The 
authors perform a number of experiments on 6 different benchmark NLP tasks and find that prompt-based instruction 
tuned LLMs outperform the classifier-head based model, while needing 100x-1000x fewer prompt data points. In my 
version of the experiment, I kept things lightweight with a rudimentary causal LM GPT2, and a simple discriminative task.

Experiment design: The experiment trains a series of models on text classification task in two training paradigms 
using the same base generative LLM. The former training setup uses the more conventional approach of fitting a 
classifier head after LLM layers, and training the classifier (with or without frozen LLM layers) to map input 
documents to a fixed set of output labels. The latter setup directly tunes (all or part of) the LLM layers to 
autoregressively generate the correct output label given a prompt containing an instruction and the input document 
text to be classified. Each model is trained only for 5 epochs.

**Experiment Setup**
- __Task__: Topic classification of news articles
- __Dataset__: ag-news ([huggingface dataset hub](https://huggingface.co/datasets/ag_news))
- __Pretrained LLM__: GPT-2 (for a `$$`-and-resource friendly experiment)

- Comment on choice of task and dataset:
  - I was inclined towards testing generative LLMs on discriminative NLP tasks because discriminative tasks are more 
    easily verifiable for accuracy than abstractive or open-ended language tasks.
    <d-footnote>Note on roads not taken:
    (1) There is no reason why this experiment could not be done on harder sequence extractive/classification tasks 
    such as extractive QA, or generative abstractive tasks. While those would have made for a more satisfying 
    experiment, I abandoned my training efforts mid-way, in favour of cheaper topic classification. (2) Purely 
    generative tasks which do not have deterministic outputs (e.g. abstractive summarisation or Q/A) were also left out of scope as they were 
    harder to evaluate objectively. Eugene Yan does a good job of documenting the landscape of challenges 
    and techniques available: https://eugeneyan.com/writing/abstractive/. 
    </d-footnote>
  - Label Predictions: To obtain a class label for input sentences in the generative case, I used a verbaliser to 
    map the output of the first 3 generated tokens from the fine-tuned LLM to map to one of the 4 class labels using 
    a scoring function such as bert-score and rouge metrics.
  - Varying the dataset size for training: I implement a custom sequential data sampler in Pytorch that increments 
    the training data volume exponentially to the base 10. I start my training runs with 10 data points, and then 
    sequentially increment the training set to 100, 1000, 10000, 100000 data points. In the reference paper, the 
    authors use a similar exponential philosophy. Additionally, they perform multiple data sampling runs to build 
    more robust training curves (I skip that aspect). 
  - Training curves: I then obtain training curves for each setup, by training 5 different models on incremental 
    segments of the labeled dataset. The classification accuracy on a held out test set is plotted for each 
    model tuned on varying training data sizes. 
  - Process for reaching a verdict: The hold-out test-set performance trends for both classifier head and the 
    prompt-based instruction tuning scenarios are compared to compute the relative data advantage either method may 
    have on the other.

  
#### Implementation notes

- I implement a custom sequential data sampler in Pytorch that increments the training data volume exponentially to the 
  base 10. I start my training runs with 10 data points, and then sequentially increment the training set to $100$, 
  1000, 10000, 100000 data points. In the reference paper, the authors use a similar exponential philosophy. 
  Additionally, they perform multiple data sampling runs to build more robust training curves. I slack off on this aspect.
- Link to code: TBD

|                       |                                                                                                 Setup 1                                                                                                  |                                                                                                                                 Setup 2                                                                                                                                 |
|:---------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|         Input         | Text of news articles. <br> Example.: "_South Korea lowers interest rates South Korea's central bank cuts interest rates by a quarter percentage point to 3.5 in a bid to drive growth in the economy._" | Text of news article embedded in a prompt template. Example: "`Given news article:` _South Korea lowers interest rates South Korea's central bank cuts interest rates by a quarter percentage point to 3.5 in a bid to drive growth in the economy._"`. The topic is:`" |
|     Target output     |                                                     Integer id corresponding to 1 of the 4 topic classes ("Sports", "World", "Business", "Sci/Tech")                                                     |                                                                                                  (Tokenised) Text of label: "Sports" , "World", "Business", "Sci/Tech"                                                                                                  |
| ML Model architecture |                                                         Pretrained model GPT2, followed by a multiclass classifier head. Only last layer trained                                                         |                                                                                            Pretrained model GPT2. All layers trainable for autoregressive output generation                                                                                             |
|  Learning Objective   |                                                                            Minimise Cross entropy loss of classifying labels                                                                             |                                                                                                   Next token prediction objective of the Causal Autoregressive model                                                                                                    |
|       Inference       |                                                                Argmax of a softmax over the output logits yields the predicted class id.                                                                 |                                                generate a sequence of upto 5 tokens that are most likely to follow the input prompt, and then use a verbaliser to map the classes to one of the 4 topic labels (or none)                                                |



<div class="row mt-3"  style="width:600px; height:400px align:right">
    <div class="col-sm mt-3 mt-md-0" >
    <figcaption> Figure 1: Test-set performance of models against number of data points tehy were trained on 
</figcaption>
        {% include figure.html path="assets/img/sample_efficiency/data_points_vs_prompts_ag_news_w_bordered.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>  

#### Experiment Observations and Outcome
- *Seeing is believing*. The above experiment plot indicates that teaching a model through next-token prediction format 
helps the pretrained model learn quicker and with lesser data -> more effective transfer learning! 
- Steep gain in model performance with just a 100 prompts! When teaching a model through prompt-response formatted 
  pairs we see that a model trained on a mere 100 instruction prompts reaches an accuracy of 50%, as compared to the more 
  traditional classifier training approach which is at accuracy of 0.06%. 
- The prompt-based learning (instruction tuning) method reaches an accuracy of over 60% when trained with 10,000 data 
points, in comparison to the mere 30% achieved by classifier-based model at the same 10,000 data point mark. If we 
  linearly interpolate the training curves for model performance on training dataset size, then the classifier based 
  model would have required 50,000 data points to achieve an accuracy of 60%.
- In conclusion: Similar to the results in the underlying paper [7], we find that in our primitive experiment, 
  prompt-based instruction tuning of models was ~5x times more sample efficient than conventional classifier-head 
  based models. In [7] the authors conduct more rigorous experiments where models are trained with better 
  hyperparameters and to convergence, across multiple runs. They infer that 'a prompt was equivalent to 100s of data 
  points' when evaluated on SUPERGLUE benchmark tasks.

 
_Experiment Limitations_ (largely due to decisions of saving on training and infrastructure costs)

|                                      Limitation                                       |                                                                                                                                                                                                                                                                                                                                   Elaboration                                                                                                                                                                                                                                                                                                                                    |
|:-------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                               Under-training of models                                |                                                                                                                                                                                                                                                        I train each model (regardless of training dataset size) for only 5 epochs. This is well below convergence. Related works often train models for > 50 epochs.[7,]                                                                                                                                                                                                                                                         |
|             Use of more primitive model GPT-2 and not powerful Causal LLM             |                                                                                                                                                                                                                      The comparative gains of the prompt-based learning (instruction tuning) method is expected to be more stark when using a powerful FM like GPT-3, or the LLama models [16][17], and could have led to more decisive experiment results                                                                                                                                                                                                                       |
|                       Choice of rudimentary classification task                       |                                                                                                                                                                                                                      I chose the simplest task of sentence classification. It is not clear how my results extrapolate to more advanced tasks of say abstractive NLP. However, the simple classification setup was sufficient for purpose of my experiment.                                                                                                                                                                                                                       |
|                   Did not use a fewshot-prompting setup as baseline                   |                                                                                                                                                                                                                                                                          I do not provide any comparison to the performance reachable by pure prompting and in-context learning (no training scenario).                                                                                                                                                                                                                                                                          |
| Evaluation of generative models is hard, and I did not build a rigourous-enough evals | Mapping the output of generative models to discriminative labels has been notorious in the community for lack of rigour. For instance, huggingface reports evaluation hurdles and holes while carrying out LLM evaluation on the benchmark task of MCQ-MMLU [15]. I used a verbaliser to map the output of the first 5 generated tokens from the fine-tuned LLM to map to one of the 4 class labels using a scoring function such as bert-score and rouge. These are not without limitations, see [here](https://eugeneyan.com/writing/llm-patterns/?utm_source=convertkit&utm_medium=email&utm_campaign=How+to+Match+LLM+Patterns+to+Problems%20-%2011495339#more-about-evals). |   


<h3 id="2-reflections"> Main reflections </h3>
Some elementary realisations that grounded my understanding of what made prompting powerful
(besides the fact that today's LLMs are trained on massive amounts of data [23] <d-footnote> This is summed up 
well in [23] as "Transfer learning makes FMs possible, but it is the scale that makes them powerful"
</d-footnote>).
* `Talk to and teach the model in its native tongue`: Prompting a model for completion in a certain task 
  during inference is like interrogating the model in the native tongue it was taught in. Thus it would be more 
effective to continue to fine-tune the model with a learning objective 
similar to its pretraining objective (e.g. next token prediction, denoising) on datasets refomatted as 
pompt-completion pairs. This may be common knowledge now, but I still have deep appreciation for this point. I 
would **strongly** recommend this excellent review paper on the different LM architectures, training 
objectives and their relationship with prompts to conduct various NLP tasks: [Pretrain, Prompt, Predict](https://arxiv.org/abs/2107.13586) [8].
 
* `All tasks can be viewed as generation`: Almost all non-generative NLP tasks that were traditionally thought of as 
  discriminative (e.g.Topic Classification, structured extraction (NER, EL, extractive QA), etc) could be posed as a 
  generative task using prompt-completion frameworks. Instead of trying to modify the method (hammer)  to work with the task (bolt), transform the task to 
  a nail instead. Some examples:
	-  Extract structured events (NER + Relation extraction) from natural language: In [5], the authors design few-shot 
       prompts in form of semi-instantiated Python classes for `Event`, with event entities and relations as 
       attributes of the class. They then prompt code-to-text FMs to complete the extraction of remainder attributes from the text 
	-  Old gem, but I found myself revisiting the T5 paper [4] which reverberates the philosophy of viewing 
	   all tasks as generation (sequence to sequence generation)
	
* `The same fine-tuning recipe behind most of the recent successful LLM apps` The recent surge of 
  successful prompt-based technologies built atop general purpose [Foundation Models (FMs)](https://hai.stanford.edu/news/what-foundation-model-explainer-non-experts), whether chat-agents, coding 
  assistants, or natural-language search tools, all inherit from the same 3-stage fine-tuning recipe as seen in Figure 2.

<div class="row mt-3"  style="width:800px; height:500px align:right">
    <div class="col-sm mt-3 mt-md-0" >
    <figcaption> Figure 2: Fine-tuning recipe of FMs </figcaption>
        {% include figure.html path="assets/img/sample_efficiency/fm_to_chatgpts.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div> 

  - The 3-stages of the fine-tuning recipe indicated above:
	- **Stage 1 -Pretrained FM Selection**: Take a foundational LLM which was pretrained at scale on 
      subject domains and data formats that are relevant to your task. In case no suitable FM is available in 
      domain of interest, one might also consider pretraining an LLM from scratch (e.g. BloomBergGPT)
    - **Stage 2 - Supervised Fine-tuning/Instruction Tuning**: Adapt all or some of FM weights by supervised 
      fine-tuning on
	  task-appropriate data points formatted as prompt-response pairs. In case of multi-task fine-tuning each task 
      dataset may be formatted with its own distinct instruction sets or prefix(e.g. FLAN-lineage of 
      models, T5). Parameter-Efficient Fine-Tuning (PEFT) methods may be used to update only part of the LLM weights 
      [9].
		- In a broad strokes, ChatGPT is the output of instruction fine-tuning GPT-3.5+ over training examples that	
		  were in format of completing conversational dialog pairs (ChatGPT also involves another layer of 
		  fine-tuning where the model output is aligned to human preferences using RLHF)
        - The T5 model was the result of taking the good old transformer architecture, pretraining it 
		  with a de-noising objective on unstructured text corpus, and then finally fine-tuning it on different 
		  tasks datasets that were converted into a format of text-to-text generation tasks with a task specific 
		  prefix [4]. More recently, FLAN-T5 was released as a more powerful LLM after applying the instruction-tuning 
		  process to T5 at scale [18].
        - Github CoPilot and OpenAI Codex utilise GPT-3 model variants that continued pretraining on code 
		  repositories and which were then fine-tuned on language-to-code mapped datasets.
    - **(Optional) Stage 3 - Reinforcement Learning from Human Feedback (RLHF)** Align models to human preferences 
      using RLHF. Ths involves another round of fine-tuning LLMs from step 2 with human feedback on preferred 
      responses amongst multiple valid generated outputs. Such alignment is usually performed to tune the model 
      to learn more abstract human notions such as safety, honesty, harmlessness, etc. A helpful overview is presented 
      in [Chip Huyen's RLHF blogpost](https://huyenchip.com/2023/05/02/rlhf.html)[10].
* Want to further fine-tune an already tuned FM to your own downstream use-case with instruction-tuning, other PEFT 
  approaches, or even RLHF? Well, you guessed correctly: prompt-response pairs are the labeled input-output data 
  pairs you need to for your training routine again.

<h3 id="3-takeaway-remarks"> Prompt engineering vs fine-tuning </h3>

A practical question faced is whether relying on in-context learning through carefully engineered prompts (without 
any training) is sufficient vs fine-tuning all or part of an LLM's weights on a task.

<div class="row mt-3"  style="width:400px; height:300px align:right">
    <div class="col-sm mt-3 mt-md-0" >
    <figcaption> Figure 3: Comparing task performance against model parameter size for various transfer-learning 
techniques
</figcaption>
        {% include figure.html path="assets/img/sample_efficiency/prompting_vs_fine_tuning.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div> 

There are a number of deciding factors in this situation: 
1. Size of the foundation LLM: Literature demonstrates that the number of model parameters is correlated with 
   improvements in prompting performance. In Figure 3 excerpted from one of the initial works in prompt tuning 
    by Lester et al [11], the authors show that while the advantages of fine-tuning over prompting are large for 
   small model sizes (<10B), this advantage is lost as model size approaches 10B.  
2. Does your task require specialised knowledge and complex reasoning? the more specific the knowledge required to 
   perform the target task, it is useful if the LLM has been pretrained on knowledge (data and tasks) similar to the 
   use-case domain. Prompting works well for simple tasks that rely on general-knowledge or are dependent on pattern 
   recognition (e.g. parsing fixed structure from text). However, complex, multi-step reasoning or niche domain 
   knowledge tasks may require fine-tuning for reliable and consistent performance. This is especially true if you 
   are not using an LLM pretrained in the same domain as your task. Hallucinations and factual consistency are known 
   weaknesses of out-of-box FMs [19][20]. 
3. Can you tune the instruction for desired output instead of tuning the model? There is a subset of PEFT methods 
   such as prompt-tuning, prefix tuning, etc, that attach learnable parameters to the model input so as to maximise 
   the probability of expected task output. This is in-between prompt engineering and fine-tuning of LLM weights and 
   might benefit your use case.
4. Do you have access to the actual model weights for fine-tuning? If you access the LLM behind an API, you cannot 
   fine tune the model, prompt or prefix.
5. What is economically and technically feasible for your application in production? Consider the production use 
   case. Relying on the LLM's in-context learning often requires very large descriptive prompts with advanced 
   patterns like few-shot examples or Chain of Thought. Attaching a long prompt template for every inference data 
   point can become very expensive in production. In such cases, if your task is very specific, then consider 
   fine-tuning as this can bake in the prompt behaviour into the training examples, and thus improve the zero-shot 
   model performance on the task. On the flipside, is it feasible to access and maintain the infrastructure required 
   for self-hosting and fine-tuning an LLM?


There is no 'golden goldilocks prompt' that dramatically improves performance on a given task.
In [7] the authors insightfully experiment with the effect of prompt variability on the LLM task performance. They 
find that the gains of any one prompt usually disappear over multiple runs, and conclude that prompt size and format 
is not a dominant hyperparameter for the LLM. <d-footnote> Note, the tasks studied in the paper are simpler, short 
form NLP tasks. Long form abstractive reasoning tasks have been shown to benefit from more sophisticated prompts. A 
useful summary of the more advanced prompting techniques and where to use them can be found in [12]. As mentioned 
earlier, there are also PEFT methods that experiment with soft tunable prompts. A recent paper [`LLMs as Optimisers`]
(https://arxiv.org/abs/2309.03409)[13] uses meta-prompts to prompt the LLM to build its own task optimising prompt </d-footnote>


__Evaluation is your true north star:__ 
In my experience, the single most effective pattern to guide development of LLM-centric applications is evaluation.
Evaluate predictions from an LLM of your choice (e.g. ChatGPT, Llama, etc) on metrics that are relevant to your 
task. Are you able to reliably (reproducibly and consistently) get the performance you require from prompt 
engineering alone? Does this performance hold as you increase the size of the test set you evaluate on? If yes, then 
in-context learning using prompting might be reasonable in your use case. If not, then you may consider other
approaches to adapt the LLM to your task.


#### Takeaways
- If you want to use a smaller model for specific tasks and domains, the current industry/academia verdict is that 
  fine-tuning can get you there faster and more reliably than prompt engineering. 
- Directly adapting an LLM's output to a task using fine-tuning with prompt-completion pairs (instruction tuning, or 
  even RLHF which incorporates human preferences into the LLM tuning objective) is much more sample efficient than 
  using conventional labeled input-output data pairs.
- Use PEFT methods to make fine-tuning LLMs more accessible and feasible. Methods such as LoRA [21] (or QLoRA[22]) are 
  incredibly universal and can be applied to both instruction-tuning and RLHF.
- Build evals as a critical part of your workflow. An evaluation routine that captures the performance criteria that 
  one most cares about is a critical aid to  deciding between approaches for LLM application.

### Acknowledgements
I am very grateful to [Corey Harper](https://www.linkedin.com/in/corey-harper-7628509/) for going above and beyond 
with his editorial review of the post, and to [Raahul Dutta](https://www.linkedin.com/in/raahuldutta/) for his 
helpful feedback. 
 
### References
[1] [Against LLM maximalism](https://explosion.ai/blog/against-llm-maximalism) by Mathhew Honnibal

[2] [Patterns for Building LL-Based systems and products](https://eugeneyan.com/writing/llm-patterns/) by Eugene Yan

[3] [Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html) by Chip Huyen

[4] Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." The 
Journal of Machine Learning Research 21.1 (2020): 5485-5551.

<!--[4] [AKBC 2022 Keynote Talks](https://wise-supervision.github.io/#Speakers)--> 

[5] Code4Struct[[arXiv]](https://arxiv.org/abs/2210.12810) 

[6] Agirre, Eneko. "Few-shot Information Extraction is Here: Pre-train, Prompt and Entail." Proceedings of the 45th 
International ACM SIGIR Conference on Research and Development in Information Retrieval. 2022.[[ACM]](https://dl.acm.org/doi/abs/10.1145/3477495.3532786)

[7] Scao, Teven Le, and Alexander M. Rush. "How many data points is a prompt worth?." arXiv preprint arXiv:2103.08493 
(2021). [[arXiv]](https://arxiv.org/abs/2103.08493)

[8] Liu, Pengfei, et al. "Pre-train, prompt, and predict: A systematic survey of prompting methods in natural 
language processing." ACM Computing Surveys 55.9 (2023): 1-35. [[ACM]](https://dl.acm.org/doi/full/10.1145/3560815)

[9] Lialin, Vladislav, Vijeta Deshpande, and Anna Rumshisky. "Scaling down to scale up: A guide to 
parameter-efficient fine-tuning." arXiv preprint arXiv:2303.15647 (2023).

[10] [RLHF](https://huyenchip.com/2023/05/02/rlhf.html) by Chip Huyen

[11] Lester, Brian, Rami Al-Rfou, and Noah Constant. "The power of scale for parameter-efficient prompt tuning." arXiv 
   preprint arXiv:2104.08691 (2021). [[arXiv]](https://arxiv.org/abs/2104.08691)

[12] [Advanced Prompt-engineering](https://towardsdatascience.com/advanced-prompt-engineering-f07f9e55fe01) by 
Cameron R. Wolfe

[13] Yang, Chengrun, et al. "Large language models as optimizers." arXiv preprint arXiv:2309.03409 (2023).

[14] Huggingface tutorial and implementation of Parameter Efficient Fine-Tuning methods: https://huggingface.
co/blog/peft

[15] Huggingface blog ["What's going on with the Open LLM Leaderboard?"](https://huggingface.co/blog/evaluating-mmlu-leaderboard)

[16] GPT understands, too: Liu, X., et al. "GPT understands, too. arXiv." arXiv preprint arXiv:2103.10385 (2021).

[17] Touvron, Hugo, et al. "Llama 2: Open foundation and fine-tuned chat models." arXiv preprint arXiv:2307.09288 (2023).

[18] Chung, Hyung Won, et al. "Scaling instruction-finetuned language models." arXiv preprint arXiv:2210.11416 (2022).

[19] Min, Sewon, et al. "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.
" arXiv preprint arXiv:2305.14251 (2023).

[20] Devaraj, Ashwin, et al. "Evaluating factuality in text simplification." Proceedings of the conference. 
Association for Computational Linguistics. Meeting. Vol. 2022. NIH Public Access, 2022.

[21] Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." arXiv preprint arXiv:2106.09685 (2021).

[22] Dettmers, Tim, et al. "Qlora: Efficient finetuning of quantized llms." arXiv preprint arXiv:2305.14314 (2023).

[23] Bommasani, Rishi, et al. "On the opportunities and risks of foundation models." arXiv preprint arXiv:2108.07258 (2021).