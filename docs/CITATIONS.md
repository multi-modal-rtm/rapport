# Section 2
## dialoguernn
<@inproceedings{10.1609/aaai.v33i01.33016818,
author = {Majumder, Navonil and Poria, Soujanya and Hazarika, Devamanyu and Mihalcea, Rada and Gelbukh, Alexander and Cambria, Erik},
title = {DialogueRNN: an attentive RNN for emotion detection in conversations},
year = {2019},
isbn = {978-1-57735-809-1},
publisher = {AAAI Press},
url = {https://doi.org/10.1609/aaai.v33i01.33016818},
doi = {10.1609/aaai.v33i01.33016818},
abstract = {Emotion detection in conversations is a necessary step for a number of applications, including opinion mining over chat history, social media threads, debates, argumentation mining, understanding consumer feedback in live conversations, and so on. Currently systems do not treat the parties in the conversation individually by adapting to the speaker of each utterance. In this paper, we describe a new method based on recurrent neural networks that keeps track of the individual party states throughout the conversation and uses this information for emotion classification. Our model outperforms the state-of-the-art by a significant margin on two different datasets.},
booktitle = {Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence and Thirty-First Innovative Applications of Artificial Intelligence Conference and Ninth AAAI Symposium on Educational Advances in Artificial Intelligence},
articleno = {837},
numpages = {8},
location = {Honolulu, Hawaii, USA},
series = {AAAI'19/IAAI'19/EAAI'19}
}>
## dialoguegcn
<@inproceedings{Ghosal2019DialogueGCNAG,
  title={DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation},
  author={Deepanway Ghosal and Navonil Majumder and Soujanya Poria and Niyati Chhaya and Alexander Gelbukh},
  booktitle={Conference on Empirical Methods in Natural Language Processing},
  year={2019},
  url={https://api.semanticscholar.org/CorpusID:201698197}
}>
## MMGCN
<@inproceedings{hu-etal-2021-mmgcn,
    title = "{MMGCN}: Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation",
    author = "Hu, Jingwen  and
      Liu, Yuchen  and
      Zhao, Jinming  and
      Jin, Qin",
    editor = "Zong, Chengqing  and
      Xia, Fei  and
      Li, Wenjie  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.440/",
    doi = "10.18653/v1/2021.acl-long.440",
    pages = "5666--5675",
    abstract = "Emotion recognition in conversation (ERC) is a crucial component in affective dialogue systems, which helps the system understand users' emotions and generate empathetic responses. However, most works focus on modeling speaker and contextual information primarily on the textual modality or simply leveraging multimodal information through feature concatenation. In order to explore a more effective way of utilizing both multimodal and long-distance contextual information, we propose a new model based on multimodal fused graph convolutional network, MMGCN, in this work. MMGCN can not only make use of multimodal dependencies effectively, but also leverage speaker information to model inter-speaker and intra-speaker dependency. We evaluate our proposed model on two public benchmark datasets, IEMOCAP and MELD, and the results prove the effectiveness of MMGCN, which outperforms other SOTA methods by a significant margin under the multimodal conversation setting."
}>
## COGMEN
<@inproceedings{joshi-etal-2022-cogmen,
    title = "{COGMEN}: {CO}ntextualized {GNN} based Multimodal Emotion recognitio{N}",
    author = "Joshi, Abhinav  and
      Bhat, Ashwani  and
      Jain, Ayush  and
      Singh, Atin  and
      Modi, Ashutosh",
    editor = "Carpuat, Marine  and
      de Marneffe, Marie-Catherine  and
      Meza Ruiz, Ivan Vladimir",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.306/",
    doi = "10.18653/v1/2022.naacl-main.306",
    pages = "4148--4164",
    abstract = "Emotions are an inherent part of human interactions, and consequently, it is imperative to develop AI systems that understand and recognize human emotions. During a conversation involving various people, a person{'}s emotions are influenced by the other speaker{'}s utterances and their own emotional state over the utterances. In this paper, we propose COntextualized Graph Neural Network based Multi- modal Emotion recognitioN (COGMEN) system that leverages local information (i.e., inter/intra dependency between speakers) and global information (context). The proposed model uses Graph Neural Network (GNN) based architecture to model the complex dependencies (local and global information) in a conversation. Our model gives state-of-the- art (SOTA) results on IEMOCAP and MOSEI datasets, and detailed ablation experiments show the importance of modeling information at both levels."
}>
## EmoShiftNet
<@ARTICLE{10.3389/frai.2025.1618698,
    
AUTHOR={Nirujan, Hinduja  and Priyadarshana, Y. H. P. P. },
           
TITLE={EmoShiftNet: a shift-aware multi-task learning framework with fusion strategies for emotion recognition in multi-party conversations},
          
JOURNAL={Frontiers in Artificial Intelligence},
          
VOLUME={Volume 8 - 2025},
  
YEAR={2025},
  
URL={https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1618698},
  
DOI={10.3389/frai.2025.1618698},
  
ISSN={2624-8212},
  
ABSTRACT={IntroductionEmotion Recognition in Conversations (ERC) is vital for applications such as mental health monitoring, virtual assistants, and human–computer interaction. However, existing ERC models often neglect emotion shifts—transitions between emotional states across dialogue turns in multi-party conversations (MPCs). These shifts are subtle, context-dependent, and complicated by class imbalance in datasets such as the Multimodal EmotionLines Dataset (MELD).MethodsTo address this, we propose EmoShiftNet, a shift-aware multi-task learning (MTL) framework that jointly performs emotion classification and emotion shift detection. The model integrates multimodal features, including contextualized text embeddings from BERT, acoustic features (Mel-Frequency Cepstral Coefficients, pitch, loudness), and temporal cues (pause duration, speaker overlap, utterance length). Emotion shift detection is incorporated as an auxiliary task via a composite loss function combining focal loss, binary cross-entropy, and triplet margin loss.ResultsEvaluations on the MELD dataset demonstrate that EmoShiftNet achieves higher overall F1-scores than both traditional and graph-based ERC models. In addition, the framework improves the recognition of minority emotions under imbalanced conditions, confirming the effectiveness of incorporating shift supervision and multimodal fusion.DiscussionThese findings highlight the importance of modeling emotional transitions in ERC. By leveraging multi-task learning with explicit shift detection, EmoShiftNet enhances contextual awareness and offers more robust performance for multi-party conversational emotion recognition.}}>
## SocialArcNet
<@inproceedings{10.1145/3789692.3789829,
author = {Xojamqulov, Abdulaziz and Atadjanov, Ibragim and Abdulali, Arsen and Pirimqulova, Zilola and Ruzimboev, Khusniddin and Muxamadiyev, Sanjar},
title = {The Social Arc: A Memory-Augmented Graph Network for Multimodal Interaction Understanding},
year = {2026},
isbn = {9798400720918},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3789692.3789829},
doi = {10.1145/3789692.3789829},
abstract = {Understanding human emotion in conversation is a complex task that requires interpreting not just the multimodal cues of a single utterance, but also the broader conversational context. Most existing models fail to capture the long-term, dynamic history of multi-party interactions, treating speakers or utterances in isolation. To address this gap, we propose SocialArcNet, a novel architecture that explicitly models the social arc of a conversation. Our model integrates powerful unimodal backbones with a recurrent Graph Neural Network (GNN) that functions as a social memory. By maintaining and updating a hidden state for each speaker as a distinct node in the graph, SocialArcNet tracks the evolving affective trajectory of each participant. We demonstrate the effectiveness of our approach, that achieves a competitive weighted F1-score of 0.62, on the MELD dataset, outperforming current baselines. Our results validate that modeling the dynamic speaker state is a crucial strategy for contextual emotion recognition. Furthermore, we highlight the critical role of advanced loss functions and regularization in overcoming the severe class imbalance and overfitting challenges inherent in this domain. Our code available at},
booktitle = {Proceedings of the 9th International Conference on Future Networks and Distributed Systems},
pages = {1069–1075},
numpages = {7},
keywords = {Affective Computing, Context Modeling, Graph Neural Networks, Multimodal Emotion Recognition},
location = {
},
series = {ICFNDS '25}
}>
## EmoBERTa
<@article{DBLP:journals/corr/abs-2108-12009,
  author       = {Taewoon Kim and
                  Piek Vossen},
  title        = {EmoBERTa: Speaker-Aware Emotion Recognition in Conversation with RoBERTa},
  journal      = {CoRR},
  volume       = {abs/2108.12009},
  year         = {2021},
  url          = {https://arxiv.org/abs/2108.12009},
  eprinttype   = {arXiv},
  eprint       = {2108.12009},
  timestamp    = {Mon, 06 Sep 2021 16:42:14 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2108-12009.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}>
## CoMPM
<@article{DBLP:journals/corr/abs-2108-11626,
  author       = {Joosung Lee and
                  Wooin Lee},
  title        = {CoMPM: Context Modeling with Speaker's Pre-trained Memory Tracking
                  for Emotion Recognition in Conversation},
  journal      = {CoRR},
  volume       = {abs/2108.11626},
  year         = {2021},
  url          = {https://arxiv.org/abs/2108.11626},
  eprinttype   = {arXiv},
  eprint       = {2108.11626},
  timestamp    = {Fri, 27 Aug 2021 15:02:29 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2108-11626.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}>
## SPCL
<@misc{song2022supervisedprototypicalcontrastivelearning,
      title={Supervised Prototypical Contrastive Learning for Emotion Recognition in Conversation}, 
      author={Xiaohui Song and Longtao Huang and Hui Xue and Songlin Hu},
      year={2022},
      eprint={2210.08713},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2210.08713}, 
}>
## MissBench
<@misc{pham2026missbenchbenchmarkingmultimodalaffective,
      title={MissBench: Benchmarking Multimodal Affective Analysis under Imbalanced Missing Modalities}, 
      author={Tien Anh Pham and Phuong-Anh Nguyen and Duc-Trong Le and Cam-Van Thi Nguyen},
      year={2026},
      eprint={2603.09874},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.09874}, 
}>
## Strong and Simple Baselines for Multimodal Utterance Embeddings
<@article{DBLP:journals/corr/abs-1906-02125,
  author       = {Paul Pu Liang and
                  Yao Chong Lim and
                  Yao{-}Hung Hubert Tsai and
                  Ruslan Salakhutdinov and
                  Louis{-}Philippe Morency},
  title        = {Strong and Simple Baselines for Multimodal Utterance Embeddings},
  journal      = {CoRR},
  volume       = {abs/1906.02125},
  year         = {2019},
  url          = {http://arxiv.org/abs/1906.02125},
  eprinttype   = {arXiv},
  eprint       = {1906.02125},
  timestamp    = {Thu, 13 Jun 2019 13:36:00 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1906-02125.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}>
## Multimodal Emotion Recognition and Sentiment Analysis in Multi-Party Conversation Contexts
<@misc{farhadipour2025multimodalemotionrecognitionsentiment,
      title={Multimodal Emotion Recognition and Sentiment Analysis in Multi-Party Conversation Contexts}, 
      author={Aref Farhadipour and Hossein Ranjbar and Masoumeh Chapariniya and Teodora Vukovic and Sarah Ebling and Volker Dellwo},
      year={2025},
      eprint={2503.06805},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.06805}, 
}>
## Tensor Fusion Network for Multimodal Sentiment Analysis
<@article{DBLP:journals/corr/ZadehCPCM17,
  author       = {Amir Zadeh and
                  Minghai Chen and
                  Soujanya Poria and
                  Erik Cambria and
                  Louis{-}Philippe Morency},
  title        = {Tensor Fusion Network for Multimodal Sentiment Analysis},
  journal      = {CoRR},
  volume       = {abs/1707.07250},
  year         = {2017},
  url          = {http://arxiv.org/abs/1707.07250},
  eprinttype   = {arXiv},
  eprint       = {1707.07250},
  timestamp    = {Thu, 25 Jul 2019 10:47:42 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/ZadehCPCM17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}>
## Are GANs Created Equal? A Large-Scale Study
<@misc{lucic2018ganscreatedequallargescale,
      title={Are GANs Created Equal? A Large-Scale Study}, 
      author={Mario Lucic and Karol Kurach and Marcin Michalski and Sylvain Gelly and Olivier Bousquet},
      year={2018},
      eprint={1711.10337},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/1711.10337}, 
}>
## On the State of the Art of Evaluation in Neural Language Models
<@article{DBLP:journals/corr/MelisDB17,
  author       = {G{\'{a}}bor Melis and
                  Chris Dyer and
                  Phil Blunsom},
  title        = {On the State of the Art of Evaluation in Neural Language Models},
  journal      = {CoRR},
  volume       = {abs/1707.05589},
  year         = {2017},
  url          = {http://arxiv.org/abs/1707.05589},
  eprinttype   = {arXiv},
  eprint       = {1707.05589},
  timestamp    = {Mon, 13 Aug 2018 16:47:01 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/MelisDB17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}>
## Improving Reproducibility in Machine Learning Research
<@article{JMLR:v22:20-303,
  author  = {Joelle Pineau and Philippe Vincent-Lamarre and Koustuv Sinha and Vincent Lariviere and Alina Beygelzimer and Florence d'Alche-Buc and Emily Fox and Hugo Larochelle},
  title   = {Improving Reproducibility in Machine Learning Research(A Report from the NeurIPS 2019 Reproducibility Program)},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {164},
  pages   = {1--20},
  url     = {http://jmlr.org/papers/v22/20-303.html}
}>
## ReZero is All You Need: Fast Convergence at Large Depth
<@article{DBLP:journals/corr/abs-2003-04887,
  author       = {Thomas Bachlechner and
                  Bodhisattwa Prasad Majumder and
                  Huanru Henry Mao and
                  Garrison W. Cottrell and
                  Julian J. McAuley},
  title        = {ReZero is All You Need: Fast Convergence at Large Depth},
  journal      = {CoRR},
  volume       = {abs/2003.04887},
  year         = {2020},
  url          = {https://arxiv.org/abs/2003.04887},
  eprinttype   = {arXiv},
  eprint       = {2003.04887},
  timestamp    = {Sat, 30 Sep 2023 10:08:23 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2003-04887.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}>
## Fixup Initialization: Residual Learning Without Normalization
<@article{DBLP:journals/corr/abs-1901-09321,
  author       = {Hongyi Zhang and
                  Yann N. Dauphin and
                  Tengyu Ma},
  title        = {Fixup Initialization: Residual Learning Without Normalization},
  journal      = {CoRR},
  volume       = {abs/1901.09321},
  year         = {2019},
  url          = {http://arxiv.org/abs/1901.09321},
  eprinttype   = {arXiv},
  eprint       = {1901.09321},
  timestamp    = {Sun, 08 Aug 2021 16:40:51 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1901-09321.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}>
## MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations
<@article{DBLP:journals/corr/abs-1810-02508,
  author       = {Soujanya Poria and
                  Devamanyu Hazarika and
                  Navonil Majumder and
                  Gautam Naik and
                  Erik Cambria and
                  Rada Mihalcea},
  title        = {{MELD:} {A} Multimodal Multi-Party Dataset for Emotion Recognition
                  in Conversations},
  journal      = {CoRR},
  volume       = {abs/1810.02508},
  year         = {2018},
  url          = {http://arxiv.org/abs/1810.02508},
  eprinttype   = {arXiv},
  eprint       = {1810.02508},
  timestamp    = {Tue, 30 Oct 2018 10:49:09 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1810-02508.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}>
## Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation
<@article{DBLP:journals/corr/abs-2110-08727,
  author       = {Shichang Zhang and
                  Yozen Liu and
                  Yizhou Sun and
                  Neil Shah},
  title        = {Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation},
  journal      = {CoRR},
  volume       = {abs/2110.08727},
  year         = {2021},
  url          = {https://arxiv.org/abs/2110.08727},
  eprinttype   = {arXiv},
  eprint       = {2110.08727},
  timestamp    = {Fri, 22 Oct 2021 13:33:09 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2110-08727.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}>
## Combining Label Propagation and Simple Models Out-performs Graph Neural Networks
<@article{DBLP:journals/corr/abs-2010-13993,
  author       = {Qian Huang and
                  Horace He and
                  Abhay Singh and
                  Ser{-}Nam Lim and
                  Austin R. Benson},
  title        = {Combining Label Propagation and Simple Models Out-performs Graph Neural
                  Networks},
  journal      = {CoRR},
  volume       = {abs/2010.13993},
  year         = {2020},
  url          = {https://arxiv.org/abs/2010.13993},
  eprinttype   = {arXiv},
  eprint       = {2010.13993},
  timestamp    = {Mon, 02 Nov 2020 18:17:09 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2010-13993.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}>
## PEFT-SER: On the Use of Parameter Efficient Transfer Learning Approaches For Speech Emotion Recognition Using Pre-trained Speech Models
<@CONFERENCE{Feng2023,
	author = {Feng, Tiantian and Narayanan, Shrikanth},
	title = {PEFT-SER: On the Use of Parameter Efficient Transfer Learning Approaches For Speech Emotion Recognition Using Pre-trained Speech Models},
	year = {2023},
	journal = {2023 11th International Conference on Affective Computing and Intelligent Interaction, ACII 2023},
	doi = {10.1109/ACII59096.2023.10388152},
	url = {https://www.scopus.com/inward/record.uri?eid=2-s2.0-85184662122&doi=10.1109%2fACII59096.2023.10388152&partnerID=40&md5=eec7af45c829a958246a332f6d0906ae},
	type = {Conference paper},
	publication_stage = {Final},
	source = {Scopus},
	note = {Cited by: 38}
}>
## DialogueLLM: Context and emotion knowledge-tuned large language models for emotion recognition in conversations
@ARTICLE{Zhang2025,
	author = {Zhang, Yazhou and Wang, Mengyao and Wu, Youxi and Tiwari, Prayag and Li, Qiuchi and Wang, Benyou and Qin, Jing},
	title = {DialogueLLM: Context and emotion knowledge-tuned large language models for emotion recognition in conversations},
	year = {2025},
	journal = {Neural Networks},
	volume = {192},
	doi = {10.1016/j.neunet.2025.107901},
	url = {https://www.scopus.com/inward/record.uri?eid=2-s2.0-105012239350&doi=10.1016%2fj.neunet.2025.107901&partnerID=40&md5=4717faad5edf31c62538d5b7964c55e9},
	type = {Article},
	publication_stage = {Final},
	source = {Scopus},
	note = {Cited by: 23}
}
# Section 3
## Menon et al., 2021
<@article{Menon2020LongtailLV,
  title={Long-tail learning via logit adjustment},
  author={Aditya Krishna Menon and Sadeep Jayasumana and Ankit Singh Rawat and Himanshu Jain and Andreas Veit and Sanjiv Kumar},
  journal={ArXiv},
  year={2020},
  volume={abs/2007.07314},
  url={https://api.semanticscholar.org/CorpusID:220525799}
}>
## Hu et al., 2022
<@article{DBLP:journals/corr/abs-2106-09685,
  author       = {Edward J. Hu and
                  Yelong Shen and
                  Phillip Wallis and
                  Zeyuan Allen{-}Zhu and
                  Yuanzhi Li and
                  Shean Wang and
                  Weizhu Chen},
  title        = {LoRA: Low-Rank Adaptation of Large Language Models},
  journal      = {CoRR},
  volume       = {abs/2106.09685},
  year         = {2021},
  url          = {https://arxiv.org/abs/2106.09685},
  eprinttype   = {arXiv},
  eprint       = {2106.09685},
  timestamp    = {Tue, 29 Jun 2021 16:55:04 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2106-09685.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}>
## SocialArcNet
<@inproceedings{10.1145/3789692.3789829,
author = {Xojamqulov, Abdulaziz and Atadjanov, Ibragim and Abdulali, Arsen and Pirimqulova, Zilola and Ruzimboev, Khusniddin and Muxamadiyev, Sanjar},
title = {The Social Arc: A Memory-Augmented Graph Network for Multimodal Interaction Understanding},
year = {2026},
isbn = {9798400720918},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3789692.3789829},
doi = {10.1145/3789692.3789829},
abstract = {Understanding human emotion in conversation is a complex task that requires interpreting not just the multimodal cues of a single utterance, but also the broader conversational context. Most existing models fail to capture the long-term, dynamic history of multi-party interactions, treating speakers or utterances in isolation. To address this gap, we propose SocialArcNet, a novel architecture that explicitly models the social arc of a conversation. Our model integrates powerful unimodal backbones with a recurrent Graph Neural Network (GNN) that functions as a social memory. By maintaining and updating a hidden state for each speaker as a distinct node in the graph, SocialArcNet tracks the evolving affective trajectory of each participant. We demonstrate the effectiveness of our approach, that achieves a competitive weighted F1-score of 0.62, on the MELD dataset, outperforming current baselines. Our results validate that modeling the dynamic speaker state is a crucial strategy for contextual emotion recognition. Furthermore, we highlight the critical role of advanced loss functions and regularization in overcoming the severe class imbalance and overfitting challenges inherent in this domain. Our code available at},
booktitle = {Proceedings of the 9th International Conference on Future Networks and Distributed Systems},
pages = {1069–1075},
numpages = {7},
keywords = {Affective Computing, Context Modeling, Graph Neural Networks, Multimodal Emotion Recognition},
location = {
},
series = {ICFNDS '25}
}>
# Section 4
## dialoguernn
<@inproceedings{10.1609/aaai.v33i01.33016818,
author = {Majumder, Navonil and Poria, Soujanya and Hazarika, Devamanyu and Mihalcea, Rada and Gelbukh, Alexander and Cambria, Erik},
title = {DialogueRNN: an attentive RNN for emotion detection in conversations},
year = {2019},
isbn = {978-1-57735-809-1},
publisher = {AAAI Press},
url = {https://doi.org/10.1609/aaai.v33i01.33016818},
doi = {10.1609/aaai.v33i01.33016818},
abstract = {Emotion detection in conversations is a necessary step for a number of applications, including opinion mining over chat history, social media threads, debates, argumentation mining, understanding consumer feedback in live conversations, and so on. Currently systems do not treat the parties in the conversation individually by adapting to the speaker of each utterance. In this paper, we describe a new method based on recurrent neural networks that keeps track of the individual party states throughout the conversation and uses this information for emotion classification. Our model outperforms the state-of-the-art by a significant margin on two different datasets.},
booktitle = {Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence and Thirty-First Innovative Applications of Artificial Intelligence Conference and Ninth AAAI Symposium on Educational Advances in Artificial Intelligence},
articleno = {837},
numpages = {8},
location = {Honolulu, Hawaii, USA},
series = {AAAI'19/IAAI'19/EAAI'19}
}>
## MMGCN
<@inproceedings{10.1145/3343031.3351034,
author = {Wei, Yinwei and Wang, Xiang and Nie, Liqiang and He, Xiangnan and Hong, Richang and Chua, Tat-Seng},
title = {MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video},
year = {2019},
isbn = {9781450368896},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3343031.3351034},
doi = {10.1145/3343031.3351034},
abstract = {Personalized recommendation plays a central role in many online content sharing platforms. To provide quality micro-video recommendation service, it is of crucial importance to consider the interactions between users and items (i.e. micro-videos) as well as the item contents from various modalities (e.g. visual, acoustic, and textual). Existing works on multimedia recommendation largely exploit multi-modal contents to enrich item representations, while less effort is made to leverage information interchange between users and items to enhance user representations and further capture user's fine-grained preferences on different modalities. In this paper, we propose to exploit user-item interactions to guide the representation learning in each modality, and further personalized micro-video recommendation. We design a Multi-modal Graph Convolution Network (MMGCN) framework built upon the message-passing idea of graph neural networks, which can yield modal-specific representations of users and micro-videos to better capture user preferences. Specifically, we construct a user-item bipartite graph in each modality, and enrich the representation of each node with the topological structure and features of its neighbors. Through extensive experiments on three publicly available datasets, Tiktok, Kwai, and MovieLens, we demonstrate that our proposed model is able to significantly outperform state-of-the-art multi-modal recommendation methods.},
booktitle = {Proceedings of the 27th ACM International Conference on Multimedia},
pages = {1437–1445},
numpages = {9},
keywords = {graph convolution network, micro-video understanding, multi-modal recommendation},
location = {Nice, France},
series = {MM '19}
}>
## EmoShiftNet
<@ARTICLE{10.3389/frai.2025.1618698,
    
AUTHOR={Nirujan, Hinduja  and Priyadarshana, Y. H. P. P. },
           
TITLE={EmoShiftNet: a shift-aware multi-task learning framework with fusion strategies for emotion recognition in multi-party conversations},
          
JOURNAL={Frontiers in Artificial Intelligence},
          
VOLUME={Volume 8 - 2025},
  
YEAR={2025},
  
URL={https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1618698},
  
DOI={10.3389/frai.2025.1618698},
  
ISSN={2624-8212},
  
ABSTRACT={IntroductionEmotion Recognition in Conversations (ERC) is vital for applications such as mental health monitoring, virtual assistants, and human–computer interaction. However, existing ERC models often neglect emotion shifts—transitions between emotional states across dialogue turns in multi-party conversations (MPCs). These shifts are subtle, context-dependent, and complicated by class imbalance in datasets such as the Multimodal EmotionLines Dataset (MELD).MethodsTo address this, we propose EmoShiftNet, a shift-aware multi-task learning (MTL) framework that jointly performs emotion classification and emotion shift detection. The model integrates multimodal features, including contextualized text embeddings from BERT, acoustic features (Mel-Frequency Cepstral Coefficients, pitch, loudness), and temporal cues (pause duration, speaker overlap, utterance length). Emotion shift detection is incorporated as an auxiliary task via a composite loss function combining focal loss, binary cross-entropy, and triplet margin loss.ResultsEvaluations on the MELD dataset demonstrate that EmoShiftNet achieves higher overall F1-scores than both traditional and graph-based ERC models. In addition, the framework improves the recognition of minority emotions under imbalanced conditions, confirming the effectiveness of incorporating shift supervision and multimodal fusion.DiscussionThese findings highlight the importance of modeling emotional transitions in ERC. By leveraging multi-task learning with explicit shift detection, EmoShiftNet enhances contextual awareness and offers more robust performance for multi-party conversational emotion recognition.}}>
## SocialArcNet
<@inproceedings{10.1145/3789692.3789829,
author = {Xojamqulov, Abdulaziz and Atadjanov, Ibragim and Abdulali, Arsen and Pirimqulova, Zilola and Ruzimboev, Khusniddin and Muxamadiyev, Sanjar},
title = {The Social Arc: A Memory-Augmented Graph Network for Multimodal Interaction Understanding},
year = {2026},
isbn = {9798400720918},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3789692.3789829},
doi = {10.1145/3789692.3789829},
abstract = {Understanding human emotion in conversation is a complex task that requires interpreting not just the multimodal cues of a single utterance, but also the broader conversational context. Most existing models fail to capture the long-term, dynamic history of multi-party interactions, treating speakers or utterances in isolation. To address this gap, we propose SocialArcNet, a novel architecture that explicitly models the social arc of a conversation. Our model integrates powerful unimodal backbones with a recurrent Graph Neural Network (GNN) that functions as a social memory. By maintaining and updating a hidden state for each speaker as a distinct node in the graph, SocialArcNet tracks the evolving affective trajectory of each participant. We demonstrate the effectiveness of our approach, that achieves a competitive weighted F1-score of 0.62, on the MELD dataset, outperforming current baselines. Our results validate that modeling the dynamic speaker state is a crucial strategy for contextual emotion recognition. Furthermore, we highlight the critical role of advanced loss functions and regularization in overcoming the severe class imbalance and overfitting challenges inherent in this domain. Our code available at},
booktitle = {Proceedings of the 9th International Conference on Future Networks and Distributed Systems},
pages = {1069–1075},
numpages = {7},
keywords = {Affective Computing, Context Modeling, Graph Neural Networks, Multimodal Emotion Recognition},
location = {
},
series = {ICFNDS '25}
}>
# Section 5
## Busso et al., 2008
<@article{Busso2008IEMOCAP:Interactiveemotionaldyadic,
 author = {Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette and Lee, Sungbok and Narayanan, Shrikanth S.},
 bib2html_rescat = {emotion},
 doi = {10.1007/s10579-008-9076-6},
 journal = {Journal of Language Resources and Evaluation},
 link = {http://sail.usc.edu/publications/files/BussoLRE2008.pdf},
 month = {dec},
 number = {4},
 pages = {335-359},
 title = {IEMOCAP: Interactive emotional dyadic motion capture database},
 volume = {42},
 year = {2008}
}>