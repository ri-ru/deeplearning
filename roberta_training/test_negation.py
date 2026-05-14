"""
Test whether the model can detect shallow negation pairs as unequivalent.
These are real examples from reformulated_full.csv where the synthetic is
just the original with 'not' inserted.

Usage:
    python test_negation.py --checkpoint runs/roberta_fewshot/checkpoints
"""
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

ID2LABEL = {0: "equivalent", 1: "unequivalent"}

NEGATION_PAIRS = [
    # ── original 3 ──────────────────────────────────────────────────────────────
    {
        "description": "Does not contain → contains (gender)",
        "explanation": "The post does not contain language that targets, incites violence against, threatens, or calls for physical damage for an individual or a group of people because of their gender. It also does not contain language that humiliates, taunts, discriminates, or insults an individual or a group of people based on their gender. Therefore the post can be classified as Normal.",
        "synthetic":   "The post contains language that targets, incites violence against, threatens, or calls for physical damage for an individual or a group of people because of their gender. It also contains language that humiliates, taunts, discriminates, or insults an individual or a group of people based on their gender. Therefore the post should not be classified as Normal.",
        "true_label":  "unequivalent",
    },
    {
        "description": "Does not contain direct attack → contains direct attacks",
        "explanation": "This post does not contain any direct attack on people based on protected characteristics. It primarily discusses the devastating events in Pittsburgh and the speaker's interaction with the Mayor and Governor. The mention of being a US Marine and traveling to different places does not target any specific group.",
        "synthetic":   "This post contains direct attacks on people based on protected characteristics and should not be classified as 'Not Hate.'",
        "true_label":  "unequivalent",
    },
    {
        "description": "Neutral → does not qualify as Neutral",
        "explanation": "This post falls under the category of Neutral because it does not direct abuse at any specific identity, affiliation, or person. It is a statement that discusses the concept of exoneration in relation to the presence or absence of evidence in a crime, without expressing any form of abuse, hate, prejudice.",
        "synthetic":   "This post does not qualify as Neutral because it directs abuse at any identity, affiliation, or person, and expresses hate, prejudice, and intolerance.",
        "true_label":  "unequivalent",
    },
    # ── 17 additional negation-flip pairs from reformulated_full.csv ────────────
    {
        "description": "Not offensive → offensive (group bias)",
        "explanation": "This post is neutral because it is not offensive towards any group of people. It does not display any bias or negativity towards any specific aspect of a group. The post simply expresses excitement and anticipation for someone to meet someone else and expects it to be a joyful experience.",
        "synthetic":   "This post is not neutral because it is offensive towards any group of people. It displays bias and negativity towards any specific aspect of a group. The post expresses excitement and anticipation for someone to meet someone else but expects it to be a negative experience.",
        "true_label":  "unequivalent",
    },
    {
        "description": "Not Hate → Hate (protected characteristics, political opinions)",
        "explanation": 'This post should be classified as "Not Hate" because it does not contain any direct attack on people based on protected characteristics. While the post criticizes and insults certain individuals, it does not target them based on their race, ethnicity, national origin, religious affiliation, sexual orientation, caste, sex, gender, gender identity, or serious disease or disability. The post primarily expresses political opinions and personal experiences without directly attacking any specific group or entity based on protected characteristics.',
        "synthetic":   'This post should be classified as "Hate" because it contains direct attacks on people based on protected characteristics. While the post criticizes and insults certain individuals, it does target them based on their race, ethnicity, national origin, religious affiliation, sexual orientation, caste, sex, gender, gender identity, or serious disease or disability. The post primarily expresses political opinions and personal experiences without directly attacking any specific group or entity based on protected characteristics.',
        "true_label":  "unequivalent",
    },
    {
        "description": "Does not contain abusive speech → does contain (ethnic origin, religion)",
        "explanation": 'This post does not contain any abusive speech targeting specific group characteristics such as ethnic origin, religion, gender, or sexual orientation. The post describes a personal experience and expresses anger towards racist graffiti, but does not attack, demean, insult, or express animosity towards any specific group. Therefore, it can be classified as "Not Hate."',
        "synthetic":   'This post does contain abusive speech targeting specific group characteristics such as ethnic origin, religion, gender, or sexual orientation. The post describes a personal experience and expresses anger towards racist graffiti, but does not attack, demean, insult, or express animosity towards any specific group. Therefore, it cannot be classified as "Not Hate."',
        "true_label":  "unequivalent",
    },
    {
        "description": "Does not contain language inciting violence → contains (religious context)",
        "explanation": "The post does not contain language that targets, incites violence against, threatens, or calls for physical damage for an individual or a group of people because of some identifying trait or characteristic. Although it discusses the killing of human beings and criticizes the country's attitude towards human life, it does not specifically target or discriminate against any religious group. Therefore, it falls into the category of Normal.",
        "synthetic":   "The post contains language that targets, incites violence against, threatens, or calls for physical damage for an individual or a group of people because of some identifying trait or characteristic. Although it discusses the killing of human beings and criticizes the country's attitude towards human life, it does not specifically target or discriminate against any religious group. Therefore, it falls into the category of Abusive.",
        "true_label":  "unequivalent",
    },
    {
        "description": "Does not contain harmful content → does contain (Rabbi post)",
        "explanation": 'The post is a message from Rabbi Avini Rabin on 9/11, encouraging people to "steady their hearts." It does not contain any harmful or offensive content and is therefore considered benign. The post specifically targets the Jewish community, as it is a message from a rabbi, but it does not express any harmful intent or use stereotypes or lewd language.',
        "synthetic":   'The post is a message from Rabbi Avini Rabin on 9/11, encouraging people to "steady their hearts." It does contain harmful and offensive content and is therefore considered hateful. The post specifically targets the Jewish community, as it is a message from a rabbi, but it expresses harmful intent and uses stereotypes and lewd language.',
        "true_label":  "unequivalent",
    },
    {
        "description": "Not Hate → Hate (shooting of Black person, race)",
        "explanation": 'This post does not contain any direct attack on people based on protected characteristics. While it mentions the shooting of a black person and a black reverend speaking, it does not attack or discriminate against any specific race or group. Instead, it discusses the need for change and expresses a positive outlook on the situation, indicating a shift in perception towards law enforcement. Therefore, this post should be classified as "Not Hate."',
        "synthetic":   'This post does contain direct attacks on people based on protected characteristics and is hateful. It mentions the shooting of a black person and a black reverend speaking, and it attacks or discriminates against specific races and groups. Instead, it discusses the need for change and expresses a negative outlook on the situation, indicating a shift in perception towards law enforcement. Therefore, this post should be classified as "Hate."',
        "true_label":  "unequivalent",
    },
    {
        "description": "Not Offensive → Offensive (phone-filter request)",
        "explanation": "This post is classified as Not Offensive because it does not contain any offensive sentiment towards anyone. It is a simple request to add a filter to toggle showing phone-verified accounts in clients, which requires a development policy update. The post does not display rudeness, disrespect, or toxicity towards any individual or group.",
        "synthetic":   "This post is classified as Offensive because it contains offensive sentiment towards anyone. It is a simple request to add a filter to toggle showing phone-verified accounts in clients, which requires a development policy update. The post does not display rudeness, disrespect, or toxicity towards any individual or group.",
        "true_label":  "unequivalent",
    },
    {
        "description": "Normal → Offensive (sports article about online abuse)",
        "explanation": 'The post should be classified as "Normal" because it does not contain language that falls into the Offensive or Hate categories. The post is simply sharing an article about Thierry Henry, a former striker, speaking on BBC Newsnight about online abuse. The content of the post is sports-related and does not involve any language that humiliates, discriminates, insults, targets, incites violence against, threatens, or calls for physical damage to any individual or group of people. Therefore, it can be classified as "Normal".',
        "synthetic":   'The post should be classified as "Offensive" because it contains language that humiliates, discriminates, insults, targets, incites violence against, threatens, or calls for physical damage to any individual or group of people. The post is sharing an article about Thierry Henry, a former striker, speaking on BBC Newsnight about online abuse. The content of the post is sports-related and does not involve any language that falls into the Offensive or Hate categories. Therefore, it cannot be classified as "Normal".',
        "true_label":  "unequivalent",
    },
    {
        "description": "Does not target specific group → targets (innate characteristics)",
        "explanation": 'The post does not target any specific group or entity based on their actual or perceived innate characteristics such as race, religion, ethnicity, nationality, gender, sexual orientation, gender identity, or disability. It discusses the issue of bigotry and hypocrisy in a general manner without specifically attacking or demeaning any particular group. Therefore, the post falls under the category of "Not Hate Speech".',
        "synthetic":   'The post targets specific groups based on their actual or perceived innate characteristics such as race, religion, ethnicity, nationality, gender, sexual orientation, gender identity, or disability. It discusses the issue of bigotry and hypocrisy in a general manner without specifically attacking or demeaning any particular group. Therefore, the post falls under the category of "Hate Speech".',
        "true_label":  "unequivalent",
    },
    {
        "description": "Derogatory but not direct attack → does contain direct attack (faggot)",
        "explanation": "This post does contain derogatory language (\"stupid\" and \"faggot\") but it does not contain a direct attack on people based on protected characteristics as defined in the Hate definition. The post is criticizing someone's response to a point and making a sarcastic comment about it, but it does not specifically target any group or individual based on their protected characteristics. Therefore, the post should be classified as Not Hate.",
        "synthetic":   "This post does contain derogatory language (\"stupid\" and \"faggot\") and does contain a direct attack on people based on protected characteristics as defined in the Hate definition. The post is criticizing someone's response to a point and making a sarcastic comment about it, but it does not specifically target any group or individual based on their protected characteristics. Therefore, the post should be classified as Hate.",
        "true_label":  "unequivalent",
    },
    {
        "description": "Does not contain direct attack → does contain (doxxing context)",
        "explanation": "This post does not contain a direct attack on people based on protected characteristics. The author is discussing being doxxed and flamed, but there are no specific references to any protected characteristics such as race, ethnicity, religion, etc. Therefore, it does not qualify as hate speech.",
        "synthetic":   "This post does contain a direct attack on people based on protected characteristics. The author is discussing being doxxed and flamed, but there are specific references to protected characteristics such as race, ethnicity, religion, etc. Therefore, it qualifies as hate speech.",
        "true_label":  "unequivalent",
    },
    {
        "description": "Not Hate → Hate (single word 'Cunt', last sentence only)",
        "explanation": 'The provided post is a YouTube video link with the word "Cunt" as the text. While the term used is offensive, it does not contain a direct attack on people based on any protected characteristics mentioned in the definition of Hate. Therefore, this post should be classified as "Not Hate."',
        "synthetic":   'The provided post is a YouTube video link with the word "Cunt" as the text. While the term used is offensive, it does not contain a direct attack on people based on any protected characteristics mentioned in the definition of Hate. Therefore, this post should be classified as "Hate."',
        "true_label":  "unequivalent",
    },
    {
        "description": "Not Offensive → Offensive (women, violence strategy)",
        "explanation": "This post should be classified as Not Offensive because it does not contain any offensive sentiment towards anyone. It discusses a strategy to address violence against women by advocating for public shame of abusers and public support for victims. The post does not engage in rudeness, disrespect, or toxicity, and it does not target or offend any specific group or individual.",
        "synthetic":   "This post should be classified as Offensive because it contains offensive sentiment towards women and promotes violence against them. It discusses a strategy to address violence against women by advocating for public shame of abusers and public support for victims. The post does not engage in rudeness, disrespect, or toxicity, and it does not target or offend any specific group or individual.",
        "true_label":  "unequivalent",
    },
    {
        "description": "Not Hate Speech → Hate Speech (sarcastic remark about men)",
        "explanation": 'The post targets men by making a sarcastic remark about them ("here go men again"). However, the language used is not bias-motivated, hostile, or malicious. It does not attack men based on their innate characteristics, such as gender. Therefore, the post qualifies as Not Hate Speech.',
        "synthetic":   'The post targets men by making a sarcastic remark about them ("here go men again"). However, the language used is not bias-motivated, hostile, or malicious. It does not attack men based on their innate characteristics, such as gender. Therefore, the post qualifies as Hate Speech.',
        "true_label":  "unequivalent",
    },
    {
        "description": "Does not contain language targeting gender → contains (gender)",
        "explanation": 'The provided post does not contain language that targets, incites violence against, threatens, or calls for physical damage for an individual or a group of people because of their gender or any other identifying trait. Therefore, it falls under the category of "Normal" as it does not qualify as either "Hate" or "Offensive" based on the given definitions.',
        "synthetic":   'The provided post contains language that targets, incites violence against, threatens, or calls for physical damage for an individual or a group of people because of their gender or any other identifying trait. Therefore, it falls under the category of "Hate" as it does qualify as "Hate" based on the given definitions.',
        "true_label":  "unequivalent",
    },
    {
        "description": "Not Offensive → Offensive (TV show, Sopranos)",
        "explanation": "The post is classified as Not Offensive because it does not contain any offensive sentiment towards anyone. It simply states that the person saw an episode of The Sopranos and was applauding Tony, which does not involve any rudeness, disrespect, or toxicity towards anyone.",
        "synthetic":   "The post is classified as Offensive because it contains offensive sentiment towards someone. It simply states that the person saw an episode of The Sopranos and was applauding Tony, which involves rudeness, disrespect, and toxicity towards someone.",
        "true_label":  "unequivalent",
    },
    {
        "description": "Does not contain language targeting race → contains (Ukraine/Russia context)",
        "explanation": 'The post does not contain any language that targets, incites violence against, threatens, or calls for physical damage for an individual or a group of people because of their race or any identifying trait or characteristic. It is simply discussing warnings and developments indicating a potential attack on Ukraine by Russia. Therefore, the post can be classified as "Normal" as it does not fall into the categories of Offensive or Hate.',
        "synthetic":   'The post contains language that targets, incites violence against, threatens, and calls for physical damage for an individual or a group of people because of their race or any identifying trait or characteristic. It is discussing warnings and developments indicating a potential attack on Ukraine by Russia. Therefore, the post cannot be classified as "Normal" as it does fall into the categories of Offensive or Hate.',
        "true_label":  "unequivalent",
    },
]


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def find_checkpoint(path: str) -> str:
    p = Path(path)
    if p.name.startswith("checkpoint-"):
        return str(p)
    candidates = sorted(p.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
    if candidates:
        return str(candidates[-1])
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    checkpoint = find_checkpoint(args.checkpoint)
    print(f"Loading model from: {checkpoint}\n")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()

    print("Pairs where the synthetic just flips 'not' — all true label = unequivalent.")
    print("A shortcut learner detects the negation token, not the meaning change.\n")
    print(f"{'#':<3} {'True':<14} {'Model':<14} {'Confidence':>10}  Match  Description")
    print("-" * 100)

    correct = 0
    for i, pair in enumerate(NEGATION_PAIRS, 1):
        inputs = tokenizer(
            pair["explanation"], pair["synthetic"],
            return_tensors="pt", truncation=True, max_length=256,
        )
        with torch.no_grad():
            logits = model(**inputs).logits[0].numpy()
        probs      = softmax(logits)
        pred_label = ID2LABEL[int(probs.argmax())]
        confidence = probs[int(probs.argmax())]
        match      = "✓" if pred_label == pair["true_label"] else "✗"
        if pred_label == pair["true_label"]:
            correct += 1
        print(f"{i:<3} {pair['true_label']:<14} {pred_label:<14} {confidence:>10.1%}  {match}     {pair['description']}")

    print("-" * 100)
    print(f"\nAccuracy: {correct}/{len(NEGATION_PAIRS)} = {correct/len(NEGATION_PAIRS):.1%}")
    print("\nNOTE: scoring 100% here does NOT prove the model is good — a simple negation")
    print("detector would also score 100%. What matters is comparing the confidence here")
    print("vs the manual labels: if confidence is similarly high on both, the model may")
    print("just be detecting 'not' tokens rather than genuine meaning differences.")


if __name__ == "__main__":
    main()
