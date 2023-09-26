---
layout: distill
title:  "How many data points is a prompt worth?"
author: Payal
date:   2023-08-25 23:46:17 +0100

---
## TL;DR

This is a simple post that journals two things. First, the long urge I have had to conduct a basic experiment to see 
for myself 'how many (training) data points a (training) prompt was worth'. Given a task we wish to train an (L)LM for, can we yield 
comparable or even more performant models with a mere fraction of the labeled data points than before, by simply 
transforming the training examples differently? I share the experiment setup, code and observations from fine-tuning 
a series of models on the rudimentary task of topic classification which illustrates the sample efficiency of instruction tuning with prompts. 

Second, it consolidates reflections on what makes prompting powerful: (1) tracing how the use of prompting an LLM 
for completion is at the heart of today's class of LLM fine-tuning methods (instruction tuning, prefix tuning, RLHF, etc).
It makes the fine-tuning objective more consistent with the pretraining objective; (2) how non-generative tasks can be creatively reformulated as 
generative tasks (3) when to consider fine-tuning over prompt engineering

Note: I'm not making a case for 'LLM-maximalism' where the perfect prompt and best-on-market LLM combo magically solves 
compound NLP tasks. On the contrary to build reliable and feasible production-grade systems with the  relatively 
untamed LLMs, I identify with (1) [LLM-Pragmatism](https://explosion.ai/blog/against-llm-maximalism) <d-footnote> 
Depending on your use-case LLM Pragmatism could call for some combination of task-composibility and 
control-flows: do not aggregate all steps to be performed by the product feature/solution into one big LLM prompt. 
Instead, decompose it into sub-tasks and delegate different components to heuristic or more explainable ML solutions 
where best suited, and LLM-generation only for certain aspects such as summarisation or intent interpretation, which 
are highly abstractive in nature. An example of this is RAG (Retrieval Augmented Generation) in search or Q/A 
applications. The decision flow of control between different components can be determined with agents or rules (Some 
good examples of systems and patterns are available in [Against LLM-Maximilism](https://explosion.
ai/blog/against-llm-maximalism), [Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html#part_2_task_composability)[1] and [Patterns for Building LL-Based systems and 
products](https://eugeneyan.com/writing/llm-patterns/?utm_source=convertkit&utm_medium=email&utm_campaign=How+to+Match+LLM+Patterns+to+Problems%20-%2011495339#collect-user-feedback-to-build-our-data-flywheel)[2]) </d-footnote> 
and [Evaluation-Driven-Development](https://eugeneyan.com/writing/llm-patterns/?utm_source=convertkit&utm_medium=email&utm_campaign=How+to+Match+LLM+Patterns+to+Problems%20-%2011495339#evals-to-measure-performance)[2] 
(EDD). (2) It is important to understand when and how to fine-tune a suitable and perhaps even 'small' model for 
one's use case.


## Diving in
The genesis of this blog post was in early November 2022 after attending an [AKBC](https://www.akbc.ws/2022/) workshop. 
During two of the [keynote talks](https://wise-supervision.github.io/#Speakers)[3] it first dawned upon me how simple, 
versatile and effective the tactic of prompting was not just for in-context learning at inference time, but also to 
further fine-tune LLMs on desired tasks. Prompts allowed the framing of tasks in a way 
that was closer to (certain) LLMs' pretraining objective (next token/sentence prediction).
Suddenly it all made sense, and I felt as though I had been looking at so many of the NLP tasks (relating to structured knowledge 
extraction and classification tasks) wrong for years. Made me think about some problems from first principles again.

But then the ChatGPT-era happened and this newfound wonder with prompts felt antiquated, followed by personal 
genAI-fatigue. However I am ready to revive this article because (1) I had to scratch the itch, even though it might 
yield little new knowledge to reader and, (2) the intuition of prompt guided learning is still fundamental in (almost 
all?) approaches to fine-tuning LLMs and can often feel hidden beneath all the abstractions of handy libraries and 
APIs around such generative applications.

The structure of the post is as follows:
1. Reflections on fine-tuning LLMs with prompts (methods, existing applications, etc)
2. Experiment on sample-efficiency of prompt-based fine-tuning Vs. conventional methods
3. Observatory Note on use of prompt engineering vs fine-tuning 

### Main reflections
Here I focus on elementary realisations that grounded my understanding of what made prompting so powerful
(other than the fact that today's FMs were trained on massive amounts of data) <d-footnote> This is summed up 
well by the quote "Transfer learning makes FMs possible, but it is the scale that makes them powerful 
INSERT reference" </d-footnote>).
* `Talk to and teach the model in its native tongue`: Prompting a model for completion in a certain task direction 
  during inference is like interrogating and communicating with the model in the native tongue it was taught in. By 
  the same intuition, continuing to teach (fine-tune) the model to 
  perform desired tasks would be more effective with a learning objective similar to what its pretraining objective 
  was (e.g. next token prediction, denoising). This would mean reformulating our task data to prompt-completion pairs 
  instead of traditional supervised datasets (input, label) we are used to. This may be common knowledge now, but I still have 
  deep appreciation for this point because it undid how I viewed a lot of ML tasks in NLP
  * I would **strongly** recommend this excellent review paper on the different LM architectures, training 
          objectives and their relationship with prompts to conduct various NLP tasks: [pretrain, prompt, predict]
    (https://arxiv.org/abs/2107.13586)[4].
 
* `All tasks can be viewed as generation`: Almost all non-generative NLP tasks that were traditionally thought of as 
  discriminative (e.g.Topic Classification, structured extraction (NER, EL, extractive QA), etc) could be posed as a generative task using prompt-completion 
  frameworks. Instead of trying to modify the method (hammer)  to work with the task (bolt), transform the task to 
  a nail instead. Some examples:
	-  Extract structured events (NER + Relation extraction) from natural language: Authors design few-shot prompts in 
	   form of semi-instantiated Python classes for `Event`, with desired entities and relations as attributes of 
	   the class. They then prompt code-to-text FMs to complete the extraction of remainder 
	   attributes from the text 
	-  Old gem, but I found myself revisiting the T5 paper which reverberates the philosophy of viewing 
	   all tasks as generation (sequence to sequence generation)
	
* `The same fine-tuning recipe behind most of the recent successful LLM apps` The recent surge of 
  successful prompt-based technologies built atop general purpose FMs, whether chat-agents, coding assistants, or 
  natural-language search tools, all inherit from the same 3-step fine-tuning recipe: 
	- **Pretrained FM Selection**: Take a foundational LLM which was pretrained at scale on 
      subject domains and data formats that are relevant to your task. In case no suitable FM is available in 
      domain of interest, one might also consider pretraining an LLM from scratch (e.g. BloomBergGPT)
    - **Supervised Fine-tuning/Instruction Tuning**: Adapt all or some of FM weights by supervised fine-tuning on
	  task-appropriate data points formatted as prompt-response pairs. In case of multi-task fine-tuning each task 
      dataset may be formatted with its own distinct instruction sets or prefix(e.g. FLAN-lineage of 
      models, T5). Parameter-Efficient Fine-Tuning (PEFT) methods may be used to update only part of the LLM weights 
      [5].
		- In a broad strokes, ChatGPT is the output of instruction fine-tuning GPT-3.5+ over training examples that	
		  were in format of completing conversational dialog pairs (ChatGPT also involves another layer of 
		  fine-tuning where the model output is aligned to human preferences using RLHF)
        - T5 was the result of taking the good old transformer architecture, pretraining it 
		  with a de-noising objective on unstructured text corpus, and then finally fine-tuning it on different 
		  tasks datasets that were converted into a format of text-to-text generation tasks with a task specific 
		  prefix. More recently, FLAN-T5 was released as a more powerful LLM after applying the instruction-tuning 
		  process to T5 at scale.
        - Github CoPilot and OpenAI Codex utilise GPT-3 models that were continued to be trained on code 
		  repositories and then fine-tuned on language-to-code mapped datasets.
      - **(Optional) Reinforcement Learning from Human Feedback (RLHF)** Align models to human preferences 
        using RLHF. Ths involves another round of fine-tuning the models from step 2 with human feedback on preferred 
        responses amongst multiple valid generated outputs. Such alignment is usually performed to tune the model 
        to learn more abstract human notions such as safety, honesty, harmlessness, etc. A helpful overview is presented 
        in [Chip Huyen's RLHF blogpost](https://huyenchip.com/2023/05/02/rlhf.html)[5].
* Want to further fine-tune an LLM to your own downstream use-case with instruction-tuning, other PEFT approaches, 
  or even RLHF? Well, you guessed correctly: prompt-response pairs are the labeled input-output data pairs you need to 
  provide your training routine again.

<div class="row mt-3"  style="width:800px; height:500px align:right">
    <div class="col-sm mt-3 mt-md-0" >
        {% include figure.html path="assets/img/sample_efficiency/fm_to_chatgpts.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div> 


### Experiment
 To quantitatively observe the sample efficiency of fine-tuning LLMs using prompt-completion pairs instead of 
 conventional supervised input-output labelled data points, I performed the following experiment inspired by the 
 paper 'How many data points is a prompt worth'[[6]](https://arxiv.org/abs/2103.08493). The authors perform a number 
 of experiments on 6 different benchmark tasks and find that prompt-based instruction tuned LLM achieve higher 
 test-performance much earlier than the clf-head based model, while needing 100x-1000x fewer prompt data points.

**Experiment Setup**
- __Task__: Topic classification of news articles
- __Dataset__: ag-news ([huggingface dataset hub](https://huggingface.co/datasets/ag_news))
- __Pretrained LLM__: GPT-2
- Comment on choice of task and dataset:
  - To maintain a `$$`-and-resource friendly experiment, I kept things lightweight with a rudimentary causal LM GPT2, 
    and a simple discriminative task. 
  - I was inclined towards testing generative LLMs on sequence extraction tasks because extractive NLP applications 
    are more easily verifiable for accuracy than abstractive or open-ended language tasks since the answer is 
    contained in the input context itself. <d-footnote>Note on roads not taken: 
    There is no reason why this experiment could not be done on harder sequence extractive/classification tasks such as extractive QA, or generative 
    abstractive tasks. While those would have made for a more satisfying experiment, I abandoned my 
    training efforts mid-way, in favour of cheaper topic classification. Purely generative tasks which do not 
    have deterministic outputs (e.g. abstractive summarisation or Q/A) were left out of scope as they were 
    harder to evaluate objectively.</d-footnote>

- The experiment trains a series of models on sequence classification-style tasks and studies the 
  difference in learning outcomes when using different training approaches of discriminative vs generative 
  language modelling on the same base LLM amd dataset. The former 
  training setup uses the more conventional approach of fitting a classifier head after the LLM 
  layers, and training the classifier (with or without frozen LLM layers) to map input documents to a fixed set 
  of output labels. The latter setup directly tunes (all or part of) the LLM layers to autoregressively generate 
  the correct  output label given a prompt containing an instruction and the input document text to be classified.
  - I then obtain training curves for each setup, by training 5 different models on incremental segments of the 
    labeled training data. The classification accuracy on a held out test set is obtained for each model tuned 
    on varying training data sizes.
  - The test-set performance trends for both classifier head and the prompt-based instruction tuning 
    scenarios are compared to compute the relative data advantage either method may have on the other.


  
|                       |                                                                                                 Setup 1                                                                                                  |                                                                                                                                 Setup 2                                                                                                                                 |
|:---------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|         Input         | Text of news articles. <br> Example.: "_South Korea lowers interest rates South Korea's central bank cuts interest rates by a quarter percentage point to 3.5 in a bid to drive growth in the economy._" | Text of news article embedded in a prompt template. Example: "`Given news article:` _South Korea lowers interest rates South Korea's central bank cuts interest rates by a quarter percentage point to 3.5 in a bid to drive growth in the economy._"`. The topic is:`" |
|     Target output     |                                                     Integer id corresponding to 1 of the 4 topic classes ("Sports", "World", "Business", "Sci/Tech")                                                     |                                                                                                  (Tokenised) Text of label: "Sports" , "World", "Business", "Sci/Tech"                                                                                                  |
| ML Model architecture |                                                         Pretrained model GPT2, followed by a multiclass classifier head. Only last layer trained                                                         |                                                                                            Pretrained model GPT2. All layers trainable for autoregressive output generation                                                                                             |
|  Learning Objective   |                                                                            Minimise Cross entropy loss of classifying labels                                                                             |                                                                                                   Next token prediction objective of the Causal Autoregressive model                                                                                                    |
|       Inference       |                                                                Argmax of a softmax over the output logits yields the predicted class id.                                                                 |                                                generate a sequence of upto 5 tokens that are most likely to follow the input prompt, and then use a verbaliser to map the classes to one of the 4 topic labels (or none)                                                |



<div class="row mt-3"  style="width:600px; height:400px align:right">
    <div class="col-sm mt-3 mt-md-0" >
        {% include figure.html path="assets/img/sample_efficiency/data_points_vs_prompts_ag_news_w_bordered.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>  

#### Experiment Observations and Outcome
- **Seeing is believing**: The above experiment plot indicates that teaching a model through next-sentence 
  prediction format helps the pretrained model learn quicker and with lesser data -> more effective transfer learning!
  - Steep gain in model performance with just a 100 prompts: When teaching a model through prompt-response formatted 
    pairs we see that a model trained on a mere 100 prompt 
    (data points formatted within an instruction prompt) reaches an accuracy of 50%, as compared to the more traditional 
    classifier training approach which is at accuracy of 0.06%. 
  - The prompt-based learning (instruction tuning) method reaches an accuracy of over 60% when trained with 10000 
    while the classifier-based model achieved 30% accuracy when trained on 10000 data points. If we linearly interpolate 
    the training curves for model performance on training dataset size, then the classifier based model would have 
    required 50,000 data points to achieve an accuracy of 60%.
  - Similar to the results in the underlying paper [4], we find that in our primitive experiment, prompt-based 
    instruction tuning of models was 5x times more sample efficient than conventional classifier-head based models. In 
    [4] the authors conduct much more rigorous experiments where models are trained with better hypereparameters and 
    to convergence, across multiple runs. They infer that 'a prompt was equivalent to a 100s of data points' when 
    evaluated on SUPERGLUE benchmark tasks. 


- __Limitations of experiment__ (mostly due to not training the best models available so as to not rake up huge 
  infrastructure costs for illustrating a point better)
  - Under-training of models: I train each model (regardless of the training dataset size) for only 5 epochs in each 
    run. This is well below convergence. Related works often train models for > $50$ epochs.[CITE]
  - We use a primitive model GPT-2. The comparative gains of the prompt-based learning (instruction tuning) method is 
  expected to be more stark when using a powerful FM like GPT-3, or the Llamma models. (link to their evaluation numbers cited in release papers)
  - Did not use a powerful state of the art causal LLM: The more powerful models such as GPT-3, LLama have been 
    established to outperform GPT-2 by large strides on several reported tasks [CITE]. USing a more powerful model 
    (an open-source LLM that could be self-hosted and trained) should have lead to more decisive experiment results.
  - I chose the simplest task of sentence classification, which is far less complex than the longer, and more 
    abstractive NLP tasks. It is not clear how my results extrapolate to more advanced tasks. However, the motive 
    of this experiment was not to prove anything, it was simply to work through the motions and demonstrate to 
    myself what is now a well-accepted ~~point~~ fact about prompts and LLM fine-tuning in the community.
  - I do not provide any comparison to the performance reachable by pure prompting and in-context learning (no 
    training).
  - evaluation of generative models is hard! Notably trying to map the output of generative models to discriminative 
  labels has been notorious in the community for lack of rigour. For instance, even huggingface reports similar 
  evaluation hurdles and holes in while carrying out LLM evaluation on the benchmark task of MCQ - MMLU [7]. I 
  used a verbaliser to map the output of the first 5 generated tokens from the fine-tuned LLM to map to one of the 4 class labels using a scoring function 
  such as bert-score and rouge. These are not without limitations, see [here](https://eugeneyan.com/writing/llm-patterns/?utm_source=convertkit&utm_medium=email&utm_campaign=How+to+Match+LLM+Patterns+to+Problems%20-%2011495339#more-about-evals). 

#### Extra Implementation Notes

- I implement a custom sequential data sampler in Pytorch that increments the training data volume exponentially to the 
  base 10. I start my training runs with 10 data points, and then sequentially increment the training set to $100$, 
  1000, 10000, 100000 data points. In the reference paper, the authors use a similar exponential philosophy. 
  Additionally, they perform multiple data sampling runs to build more robust training curves. I slack off on this aspect.

- Link to code: TBD
- `#FIXME: Q: Do i need to share more notes on evals? I can simply point to code.`

  

### Prompt engineering vs fine-tuning 

A practical question faced is whether relying on in-context learning through carefully engineered prompts (without 
any training) is sufficient vs fine-tuning all or part of an LLM's weights on a task.


<div class="row mt-3"  style="width:400px; height:300px align:right">
    <div class="col-sm mt-3 mt-md-0" >
        {% include figure.html path="assets/img/sample_efficiency/prompting_vs_fine_tuning.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div> 

There are a number of deciding factors in this situation are 
1. Size of the foundation LLM: Literature demonstrates that the number of model parameters is correlated with 
   improvements in prompting performance. In Figure 3 excerpted from one of the initial works in prompt tuning 
   [Lester et al 2021][8], the authors show that while the advantages of fine-tuning over prompting are large for small 
   model sizes (<10B), this advantage is lost as model size approaches 10B. With these massive models prompting is 
   sufficient for the LLM to understand the instructions and complete the task as per expectations, and engineering 
   performance becomes comparable to that of fine-tuning and prompt-tuning.

2. Does your task require specialised knowledge and complex reasoning? the more specific the 
   knowledge required to perform the target task, it is useful if the LLM has been pretrained on knowledge (data and tasks) similar to the 
   use-case domain. Prompting works well for simple tasks that rely on general-knowledge or are dependent on pattern 
   recognition (e.g. parsing fixed structure from text). However, complex, multi-step reasoning or 
   niche domain 
   knowledge tasks may require fine-tuning for reliable and consistent performance. This is especially true if you 
   are not using an LLM pretrained in the same domain as your task. 

3. Can you tune the instruction for desired output instead of tuning the model? There is a subset of PEFT methods 
   such as prompt-tuning, prefix tuning, etc, that attach learnable parameters to 
   the model input so as to maximise the 
   probability of expected task output. This is in-between prompt engineering and 
   fine-tuning of LLM weights and might benefit your use case

4. Do you have access to the actual model weights for fine-tuning? If you access the LLM behind an API, you cannot 
   fine tune the model, prompt or prefix.

5. What is economically and technically feasible for your application in production? Consider the production use case.
   Relying on the LLM's in-context learning often requires very large descriptive prompts with advanced patterns like 
   few-shot examples or Chain of Thought. Attaching a long prompt template for every inference data point can become very 
   expensive in production. In such cases, if your task is very specific, then consider fine-tuning as this can 
   bake in the prompt behaviour into the training examples, and thus improve the zero-shot model performance on the 
   task. On the flipside, is it feasible to access and maintain the infrastructure required for self-hosting and 
   fine-tuning an LLM?

__Evaluation is your true north star:__ 
In my experience, the single most effective pattern to guide development of LLM-centric applications is evaluation. 
Evaluate predictions from an LLM of your choice (e.g. ChatGPT, Llama, etc) on metrics that are relevant 
to your task - are you able to reliably (reproducibly and consistently) get the performance you 
require from prompt engineering alone? Does this performance hold as you increase the size of the test set you 
evaluate on? If yes, then you in-context learning using prompting might be reasonable in your use case. If not, then 
you may consider approaches to adapt the LLM to your task.

But wait, there must be a perfect goldilocks prompt waiting to be engineered that solves my complete task...
I think not. There might not be a 'golden prompt' that 
dramatically improves performance on your given task. <d-footnote> In [6] the authors insightfully experiment with the 
effect of prompt variability on the LLM task performance. They find that the gains of any one prompt usually disappear over multiple 
runs, and conclude that prompt size and format is not a dominant hyperparameter for the LLM. Note, the tasks studied 
in the paper are simpler, short form NLP tasks. Long form abstractive reasoning tasks have been shown to benefit 
from more sophisticated prompts. A useful summary of the more advanced prompting techniques and where to use them 
can be found in [10]. </d-footnote>


#### Takeaways:
- If you want to use a smaller model for specific tasks and domains, the current industry/academia verdict is that 
  fine-tuning can get you there faster and more reliably than prompt engineering
- Directly adapting an LLM's output to a task using fine-tuning with prompt-completion pairs (instruction tuning, or 
  even RLHF which incorporates human preferences into the LLM tuning objective) is much more sample efficient than 
  using conventional labeled input-output data pairs
- Use PEFT methods to make fine-tuning LLMs more accessible and feasible. Methods such as LoRA (or QLoRA) are 
  incredibly universal and can be applied to both instruction-tuning and RLHF.
- Build evals as a critical part of your workflow. An evaluation routine that captures the performance criteria that 
  one most cares about is a critical aid to  deciding between approaches for LLM application.

 
### References:

[1] [Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html) by Chip Huyen

[2] [Patterns for Building LL-Based systems and products](https://eugeneyan.com/writing/llm-patterns/) by Eugene Yan

[3] [AKBC 2022 Keynote Talks](https://wise-supervision.github.io/#Speakers) 

[4] Liu, Pengfei, et al. "Pre-train, prompt, and predict: A systematic survey of prompting methods in natural 
language processing." ACM Computing Surveys 55.9 (2023): 1-35. [[ACM]](https://dl.acm.org/doi/full/10.1145/3560815)

[5] [RLHF by Chip Huyen](https://huyenchip.com/2023/05/02/rlhf.html)

[6] Scao, Teven Le, and Alexander M. Rush. "How many data points is a prompt worth?." arXiv preprint arXiv:2103.08493 
(2021). [[arXiv]](https://arxiv.org/abs/2103.08493)

[7] [What's going on with the Open LLM Leaderboard?](https://huggingface.co/blog/evaluating-mmlu-leaderboard)

[8] Lester, Brian, Rami Al-Rfou, and Noah Constant. "The power of scale for parameter-efficient prompt tuning." arXiv 
   preprint arXiv:2104.08691 (2021). [[arXiv]](https://arxiv.org/abs/2104.08691)
[9] GPT understands, too: Liu, X., et al. "GPT understands, too. arXiv." arXiv preprint arXiv:2103.10385 (2021).

[10] [Advanced Prompt-engineering](https://towardsdatascience.com/advanced-prompt-engineering-f07f9e55fe01) by 
Cameron R. Wolfe

11. Huggingface tutorial and implementation of Parameter Efficient Fine-Tuning methods : https://huggingface.
    co/blog/peft
