\documentclass[10pt,conference,compsocconf]{IEEEtran}

\usepackage{hyperref}
\usepackage{graphicx}	% For figure environment
\usepackage{xcolor}

% 1. Load the package
\usepackage[style=authoryear, backend=biber]{biblatex}

% 2. Specify the bibliography file
\addbibresource{references.bib}



\begin{document}
\title{Training Robust LLM Evaluators via Ground-Truth Adversarial Simulation
}

\author{
  Some very nice people, person 2, person 3\\
  \textit{EPFL, EE-559: Deep Learning}
}

\maketitle

\begin{abstract}

A very nice abstract


\end{abstract}
\section{Introduction}
Evaluating hate speech detection models remains a challenge due to the
subjective nature of "harm" and the high cost of human annotation. We
propose a framework to develop a high-fidelity, inexpensive "LLM-as-a-
Judge" by training it on a specialized evaluation dataset. This dataset is
constructed by pairing social media content with its existing ground-truth
labels alongside generated adversarial labels designed to simulate common
model errors. By training the judge to distinguish between these categories,
we create a tool that aligns with specific human beliefs and norms. This
method allows developers to leverage their own small-scale, manually
curated ground truths to bootstrap a scalable evaluator, reducing reliance
on zero-shot LLM judgements which often struggle with nuanced
moderation tasks.

\section{Literature overview}
\subsection{Main categories of hate speech detection techniques}
There are two classical methods to tackle hate speech detection:
\begin{itemize}
    \item \textbf{LLM-based techniques}, which tend to misclassify benign content as hateful due to over-sensitivity toward certain groups of topics (\cite{zhang2024}). Moreover, a stream of research has been focused on the generation of counter-speech to lower incivilities (\cite{hong2024outcome}). 
    \item \textbf{Few-shot techniques}, which typically function as black boxes and which are analyzed with probabilistic or inference approaches. 
\end{itemize}

\subsection{Biases in LLM-as-a-Judge models}
\cite{zheng2023judging} was the first to describe the biases encountered by LLM-as-a-Judge models that are evaluated during our project. Namely, these include: 
\begin{itemize}
    \item \textbf{Length bias:} the models tend to favor verbose answers; 
    \item  \textbf{Position bias:} the models preferred often preferred the first response in pairwise comparisons;
    \item  \textbf{Self-enhancement bias:} a slight tendency to favor the model's own outputs. 
\end{itemize}

\subsection{LLM-powered explainable hate speech detection (TARGE)}
Our pipeline was inspired by the TARGE paper (\cite{targe2025}), which uses Mistral-7B to develop transparent hate speech detection systems. This paper provides the framework where LLMs generate explicit rationales by identifying and analyzing critical textual features indicative of hate speech. 

First, the paper uses a pre-trained instruction-tuned LLM as a feature extractor. This LLM is then prompted to analyze input text and extract relevant hate speech features. Extracted rationales are then compared to references using overlap similarity at the token level and cosine similarity in latent space. If either of these indicators is of poor quality, the prompting is repeated. This iterative process helps tackle the issue of hallucinations with that small-scale LLMs may suffer from. 

Second, the original text is processed by a DistilBERT model. TARGE extracts the embedding of the final layer of the so-called [CLS] token, which serves as a summary representation of an entire sequence. In parallel, text and rationales are fed into a pre-trained RoBERTa model, where the final layer of the  [CLS] token is extracted. This second model is then kept frozen during training and therefore corresponds to a contextual encoder. 

Finally, the text embeddings from DistilBERT and feature embeddings from RoBERTa are concatenated. This combined embedding is then fed into a forward MLP (which uses ReLU as an activation function to avoid overfitting). The MLP is then trained to minimize the batch-wise binary cross-entropy loss in order to perform the final classification.




\section{Methodology}
\subsection{1. OG LLM-as-a-Judge} 
\cite{zheng2023judging})

Introduced pairwise comparison and single-answer grading using GPT-4 as an evaluator $\rightarrow$ >80\% agreement with human preferences. 

Coined the terminology + established core methodology that other work builds upon.

\subsection{More important papers } 

THE BELOW 3 PAPERS HAVE THE FOLLOWING TAKEAWAYS: 

Known biases in LLM judges:

--  length bias (preferring verbose outputs)

--position bias (favoring the first response in pairwise comparisons)

-- self-model bias. 

$\rightarrow$ approach of training on ground-truth with adversarial labels  addresses  issues by anchoring the judge in explicit human standards rather than relying on zero-shot prompting.

------
\cite{gu2024survey} organizes strategies for building reliable judge systems, covering consistency, bias mitigation, and adaptation to diverse assessment scenarios.


\cite{li2024generation} provides a complementary perspective focused on judging attributes, methodologies, applications, and limitations 


\cite{haitao2024llms} organizes the field across functionality, methodology, application, meta-evaluation, and limitation dimensions.
 


\subsection{Training Fine-Tuned LLM Judges} 
\cite{kim2023prometheus}
\cite{kim2024prometheus2}


Most prominent work specialized LLM judges is the Prometheus series.

Prometheus 1 (arXiv 2310.08491) fine-tuned LLaMA-2-Chat-13B on the "Feedback Collection," a synthetic dataset of 1K score rubrics, 20K instructions, and 100K responses with feedback generated by GPT-4.

Prometheus 2 (arXiv 2405.01535, Kim et al., 2024) extended this to support both direct assessment and pairwise ranking. Built on Mistral-7B and Mixtral-8x7B, it merged weights of models separately trained on the Feedback Collection and a new Preference Collection. It achieved the highest correlation with human evaluators across four direct assessment benchmarks.

$\rightarrow$ Two papers that showed small, open-source models can match proprietary judges when trained on high-quality rubric-based data.

Issue: another study showed that removing the reference answer caused the greatest performance drop, suggesting the model relies heavily on fuzzy matching rather than deep evaluation capability.

Therefore, using adversarial judges could make more robust judges that rely less on surface-level reference matching.

% "Fine-tuned judges degenerate into task-specific classifiers..."
\subsection{Limitations of Fine-Tuned Judges} 
\cite{huang2024empirical}
compared four fine-tuned judges (JudgeLM, PandaLM, Auto-J, Prometheus) to GPT-4. fine-tuned judges essentially degenerate into task-specific classifiers overfitted to their training distributions. They achieve strong in-domain scores but fail on fine-grained aspects like factuality (HaluEval), toxicity (ToxicChat), and safety (SALAD-Bench). Even chain-of-thought training did not improve generalization.

$\rightarrow$ By training on adversarial labels that simulate common model errors, you force the judge to learn the boundary between correct and incorrect classifications rather than memorizing surface patterns. This could yield better out-of-distribution generalization.


\subsection{Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical Lessons} 
\cite{hu2025themis}

What is currently being done: what should NOT be done. 

Standard way to train judges now: you take GPT-4, have it evaluate a bunch of stuff, then train a smaller model to copy GPT-4's judgments. It works okay, but the problem is your small model just learns to mimic GPT-4's opinions, including GPT-4's blind spots and biases. "this is what everyone else does" baseline that you're arguing against.

We do smth different: anchoring in human ground truth and teaching the model what errors look like. 


\subsection{Self-Improving Judges Without Human Annotations} 

Maybe irrelevant?  they proved you can build a decent evaluator from synthetic data alone, without any human preference labels. 






\section{Our steps to take}

1. Take social media posts that already have human-annotated hate speech labels

2. you generate "wrong but plausible" labels using an LLM

3. you train a small model to tell the difference between the real labels and the fake ones.

\subsection{Needs}

1. Ground Truths: a hate speech dataset where humans have already labeled posts. Good candidates: ToxiGen (274K samples, 13 minority groups), ToxicChat (10K real user-AI interactions); Jigsaw/Civil Comments dataset (1.8M comments from Wikipedia talk pages, CC0 license). 

--- Toxigen comment: ALICE (Adversarial classifier-in-the-loop constrained decoding): a framework that uses a classifier-in-the-loop during GPT-3 generation to produce text that is implicitly toxic but fools existing classifiers. Fine-tuning a toxicity classifier on ToxiGen data substantially improved its performance on human-written data.

2. Adversarial label generator. We prompt a large LLM to make plausible but wrong labels for each post. They need to simulate the mistakes real models make. Eg. labeling sarcastic hate speech as benign, flagging neutral mentions of minority groups as toxic, missing implicit hate that uses metaphor, or over-flagging profanity that isn't actually hateful. Ie design prompts that say something like "classify this post, but you tend to miss sarcasm" or "classify this post, but you're overly sensitive to group mentions." Each prompt profile generates a different error mode.

(SIDENOTE: SYNTHETIC DATA GENERATOR PAPERS:)

3.2 AIHSG: LLM-Generated Adversarial Implicit Hate Speech
Zhang et al. (2025), "Enhancing implicit hate speech detection via LLM-generated adversarial samples" (Int. J. Syst. Assur. Eng. Manag.), proposes a framework that uses LLMs to generate adversarial implicit hate speech samples that convey hateful intent through context and metaphor without explicit markers. These are used to augment training data for classifiers. This is a close parallel to your approach of using LLMs to generate adversarial labels rather than adversarial text.

3.3 LLM Synthetic Generation for Content Moderation
A 2025 paper in Computing (Springer) presents an automatic text data augmentation method using LLMs with demonstration-based generation for hate speech detection. The approach generates diverse samples while maintaining original writing style, bridging semantic gaps that limit traditional augmentation techniques. This work validates that LLM-generated training data can meaningfully improve hate speech classifier generalization.

3.A training dataset of (post, label, real/adversarial) triples.You pair each post with its ground-truth label and tag it "correct," then pair the same post with each adversarial label and tag it "adversarial." So for one post you might have: the post + "toxic" (ground truth, tagged correct), the post + "benign" (generated by the sarcasm-blind profile, tagged adversarial), the post + "toxic" (generated by the over-sensitive profile, tagged correct by coincidence, you'd filter these out or handle them carefully).

4. A small model to train as the judge
Fine-tune something like Mistral-7B, Llama-3-8B, or even a smaller model. The training task is: given a post and a proposed label, predict whether the label is correct or adversarial. The model learns to detect the gap between what a correct label looks like and what a plausible error looks like, for each type of content.

5. An evaluation that proves generalization
This is what makes or breaks the paper. You need to show that your judge generalizes better than baselines. Concretely, train on one dataset (say Jigsaw) and evaluate on a different one (say ToxicChat or HateBencH). Compare against Prometheus in zero-shot, against a vanilla fine-tuned classifier, and against GPT-4 as a judge. If your adversarially-trained judge outperforms on the out-of-distribution test set, you've validated the thesis.

\subsection{Experiments to run}

The first experiment is an ablation on adversarial label diversity. Train one judge with random label flips versus one with LLM-generated plausible errors. If the LLM-generated errors produce a better judge, that proves the "simulate real error modes" idea works.

Second, an out-of-distribution generalization test. Train on Jigsaw, test on ToxicChat and ToxiGen's human-annotated subset. Compare against Prometheus and zero-shot GPT-4.

Third, a cost analysis. Track how many ground-truth labels you need and how many LLM calls to generate adversarial labels. If you can bootstrap a strong judge from, say, 5K human-labeled examples plus cheap adversarial generation (for AmelioRatio). 







\section{OTHER}


3.4 HateBencH: Benchmarking Detectors on LLM-Generated Content
Shen et al. (USENIX Security 2025), "HateBencH," manually annotated 11,016 LLM-generated samples and evaluated hate speech detectors against adversarial attacks.

The most potent adversarial attack achieved a 0.966 attack success rate. This underscores both the vulnerability of current detectors and the need for adversarially robust evaluation, directly motivating your project.

4. Content Moderation and Safety Classifiers
4.1 WildGuard (Han et al., NeurIPS 2024)
WildGuard is a 7B-parameter open moderation tool trained on WildGuardMix (92K labeled examples covering vanilla and adversarial prompts across 13 risk categories). It achieves three tasks simultaneously: prompt harmfulness classification, response harmfulness detection, and refusal identification. WildGuard matches or exceeds GPT-4 on multiple benchmarks, with up to 3.9% improvement on prompt harmfulness identification and reduces jailbreak attack success from 79.8% to 2.4%.

Key methodological insight: WildGuard's success comes primarily from its training data (WildGuardTrain), not the base model choice. Testing with Llama2-7B, Llama3-8B, and Mistral-7B showed minimal variation, emphasizing that data quality is the primary driver. This strongly supports your approach of focusing on training data construction.

4.2 Llama Guard (Meta, 2023–2025)

Meta's Llama Guard series treats content moderation as an instruction-following task, fine-tuning Llama models to classify content as safe/unsafe against a configurable taxonomy. The key innovation is that safety categories can be redefined by editing the prompt, enabling zero-shot adaptation to new policies. Llama Guard 3 (8B) is multi-modal and multi-lingual. However, WildGuard demonstrated that Llama Guard lags behind on adversarial inputs and nuanced refusal detection, which are areas where your adversarial training approach could excel.

4.3 BingoGuard: Severity-Aware Moderation (ICLR 2025)

BingoGuard (ICLR 2025) argues that binary safe/unsafe classification is insufficient and proposes severity-level moderation. It fine-tunes specialized LLM response generators on seed sets of different severity levels (~300 examples each), then iteratively trains a moderator using misclassified examples. This iterative adversarial training loop is conceptually aligned with your framework of using adversarial labels to train more nuanced evaluators.




\section{Vero Milestone }

TODO: 

run API credit test for synthetic data on different models (anthropic, meta, openai) + figure out DeBERTa and do a small test with it 

POST MORTEM: 

Problems: 
-- train on Jigsaw, test on ToxiGen (original plan) 
 
 But hatecot already includes ToxiGen.
 
 Propose: train on {dynahate, cad,  measure, socialbias, toraman}, test on {toxigen, gab, anatomy}. (diff datasets) 

 --Cited paper uses DistilBERT[CLS] + RoBERTa[CLS] concatenated → MLP . We use Roberta (changed from deberta as roberta didnt work). if we want to be "TARGE-inspired" then we need to stick with what they did. 

==========
GENERATION: 

 Issues: 

 Adversarial prompts aren't aggressive enough esp for profanity. 5-30%. In total. 
 
 "profanity_fixated"  profile only flipped 5–10%

Anthropic refuses ~30% of toxic_explicit synth requests

  to do: Agree on the schema NOW: post_id, 
  gt_label, profile, provider, model, adv_label, raw, in_tok, out_tok.  



  =======
   1. DeBERTa-v3 hit NaN on the loss and anyway we should switch to roberta. 

   
  2. First full run out of memory at batch 16 / max_length 256 → dropped to max_length 128.
  
  3. Full 2000-sample / 3-epoch training: test F1 0.745, acc 0.740. ~6 min wall  time on M4.


  ====

  extra

    - 8GB  : deberta-v3-base, batch 4, max_length 128 
    
  - 16GB  : deberta-v3-base, batch 8–16, max_length 256 comfy .        
  
  - 24GB  : ecven try deberta-v3-large with batch 4.  


DATA ANOTATION

  
\begin{tabular}{|l|l|l|l|l|}
\hline Model & Calls & In tok & Out tok & USD \\
\hline claude-haiku-4-5 & 80 & 8,888 & 4,029 & \$0.029 \\
\hline claude-sonnet-4-6 & 80 & 8,968 & 3,675 & \$0.082 \\
\hline gpt-40-mini & 80 & 8,652 & 2,033 & \$0.003 \\
\hline gpt-40 & 80 & 8,652 & 2,010 & \$0.042 \\
\hline
\end{tabular}

for 15k: 

claude-haiku-4-5	$5.44
claude-sonnet-4-6	$15.38
gpt-40-mini	$0.56
gpt-40	$7.88


for synthetic data generation:

│ Synthetic generation (16 calls, 65 usable rows) │ $0.031 │ ~30 s  

\begin{table}[h]
\centering
\caption{15,000 Queries Cost Comparison by Model}
\label{tab:cost_comparison}
\begin{tabular}{lcccc}
\toprule
Model & Input Price (/MTok) & Output Price (/MTok) & Cost per 1,000 Queries & Cost per 15,000 Queries \\
\midrule
OpenAI GPT-4o-mini & \$0.15 & \$0.60 & \$1.94 & \$29.06 \\
OpenAI GPT-4o & \$2.50 & \$10.00 & \$32.25 & \$483.75 \\
Anthropic Haiku 4.5 & \$1.00 & \$5.00 & \$14.50 & \$217.50 \\
Anthropic Sonnet 4.6 & \$3.00 & \$15.00 & \$43.44 & \$651.56 \\
xAI Grok 4.20 & \$2.00 & \$6.00 & \$27.42 & \$411.30 \\
\bottomrule
\end{tabular}
\end{table}


 \begin{figure}
     \centering
     \includegraphics[width=1\linewidth]{image.png}
     \caption{Enter Caption}
     \label{fig:placeholder}
 \end{figure}

 Anthropic be like: I cannot comply with this request

 OpenAI: Those immigrants are ruining our country. We need to get rid of them all!

 Idea::::  i can try to hack the models to uncensor them. 

 https://github.com/p-e-w/heretic


 ==== 

 TODO
-- find the datasets based on the papers: Dimitri until Monday evening 
-- Flavia: generate the full dataset. 
-- Vero: go to class and find out if the project is okay 

 
 REPORT STUFF 
 reformulation can be 

 
 




\printbibliography

%\bibliographystyle{IEEEtran}
%\bibliography{literature}

% After your conclusion and before \end{document}

\appendix


\end{document}

