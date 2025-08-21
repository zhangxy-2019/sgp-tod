# SGP-TOD

### Overview

The schema data and main scripts have been uploaded, with a detailed organizational structure to be released shortly. Please cite us if the open-sourced files are helpful to you.

### Citation

```bibtex
@inproceedings{zhang-etal-2023-sgp,
    title = "{SGP}-{TOD}: Building Task Bots Effortlessly via Schema-Guided {LLM} Prompting",
    author = "Zhang, Xiaoying and Peng, Baolin and Li, Kun and Zhou, Jingyan and Meng, Helen",
    editor = "Bouamor, Houda and Pino, Juan and Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.891",
    doi = "10.18653/v1/2023.findings-emnlp.891",
    pages = "13348--13369",
    abstract = "Building and maintaining end-to-end task bots using minimal human effort is a long-standing challenge in dialog research. In this work, we introduce SGP-TOD, Schema-Guided Prompting for building Task-Oriented Dialog systems effortlessly based on large language models (LLMs). Utilizing the predefined task schema, i.e., belief instruction and dialog policy, we instruct fixed LLMs to generate appropriate responses on novel tasks, without the need for training data. Specifically, SGP-TOD comprises three components: an LLM for interacting with users, a Dialog State Tracking (DST) Prompter to aid the LLM in tracking dialog states with the given belief instruction, and a Policy Prompter to direct the LLM to generate proper responses adhering to the provided dialog policy. Experimental results on Multiwoz, RADDLE, and STAR datasets show that our training-free strategy, SGP-TOD, yields state-of-the-art (SOTA) zero-shot performance, significantly surpassing the few-shot approaches. In a domain-extension setting, SGP-TOD aptly adapts to new functionalities by merely adding supplementary schema rules. We make our code and data publicly available."
}
