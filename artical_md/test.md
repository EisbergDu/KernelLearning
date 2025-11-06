# Learning conditional distributions on continuous spaces

Cyril Bénézet*†  
CYRIL.BENEZET@ENSIIE.FR  
Université Paris-Saclay, CNRS, Univ Evry, ensIIE  
Laboratoire de Mathématiques et Modélisation d'Evry,  
91037, Evry-Courcouronnes, France

Ziteng Cheng*†  
ZITENG.CHENG@UTORONTO.CA  
Financial Technology Thrust  
The Hong Kong University of Science and Technology (Guangzhou)  
Guangzhou, 511400, China

Sebastian Jaimungal*§  
Department of Statistical Sciences  
University of Toronto  
Toronto, ON M5G 1Z5, Canada  
SEBASTIAN.JAIMUNGAL@UTORONTO.CA

Editor: Maxim Raginsky

## Abstract

We investigate sample-based learning of conditional distributions on multi-dimensional unit boxes, allowing for different dimensions of the feature and target spaces. Our approach involves clustering data near varying query points in the feature space to create empirical measures in the target space. We employ two distinct clustering schemes: one based on a fixed-radius ball and the other on nearest neighbors. We establish upper bounds for the convergence rates of both methods and, from these bounds, deduce optimal configurations for the radius and the number of neighbors. We propose to incorporate the nearest neighbors method into neural network training, as our empirical analysis indicates it has better performance in practice. For efficiency, our training process utilizes approximate nearest neighbors search with random binary space partitioning. Additionally, we employ the Sinkhorn algorithm and a sparsity-enforced transport plan. Our empirical findings demonstrate that, with a suitably designed structure, the neural network has the ability to adapt to a suitable level of Lipschitz continuity locally. For reproducibility, our code is available at https://github.com/zcheng-a/LCD_kNN.

**Keywords**: non-parametric statistics, Wasserstein distance, deep learning, Lipschitz continuity

## 1. Introduction

Learning the conditional distribution is a crucial aspect of many decision-making scenarios. While this learning task is generally challenging, it presents unique complexities when explored in a continuous space setting. Below, we present a classic example (cf. Booth et al. (1992); Pflug and Pichler (2016)) that highlights this core challenge.

For simplicity, we suppose the following model

$$
Y = \frac {1}{2} X + \frac {1}{2} U,
$$

where the feature variable $X$ and the noise $U$ are independent Uniform([0,1]), and $Y$ is the target variable. Upon collecting a finite number of independent samples $\mathcal{D} = \{(X_m,Y_m)\}_{m=1}^M$, we aim to estimate the conditional distribution of $Y$ given $X$. Throughout, we treat this conditional distribution as a measure-valued function of $x$, denoted by $P_x$. A naive approach is to first form an empirical joint measure

$$
\hat {\psi} := \frac {1}{M} \sum_ {m = 1} ^ {M} \delta_ {(X _ {m}, Y _ {m})},
$$

where $\delta$ stands for the Dirac measure, and then use the conditional distribution induced from $\hat{\psi}$ as an estimator. As the marginal distribution of $X$ is continuous, with probability 1 (as $\mathbb{P}(X_m = X_{m'}) = 0$ for all $m \neq m'$), we have that

$$
\widehat {P} _ {x} = \left\{ \begin{array}{l l} \delta_ {Y _ {m}}, & x = X _ {m} \text { for some } m, \\ \text {Uniform} ([ 0, 1 ]), & \text {otherwise}. \end{array} \right.
$$

Regardless of the sample size $M$, $\widehat{P}_x$ fails to approximate the true conditional distribution,

$$
P _ {x} = \mathrm {Uniform} \left([ x, x + \frac {1}{2} ]\right), \quad x \in [ 0, 1 ].
$$

Despite the well-known convergence of the (joint) empirical measure to the true distribution Dudley (1969); Fournier and Guillin (2015), the resulting conditional distribution often fails to provide an accurate approximation of the true distribution. This discrepancy could be due to the fact that calculating conditional distribution is an inherently unbounded operation. As a remedy, clustering is a widely employed technique. Specifically, given a query point $x$ in the feature space, we identify samples where $X_{m}$ is close to $x$ and use the corresponding $Y_{m}$'s to estimate $P_{x}$. Two prominent methods within the clustering approach are the kernel method and the nearest neighbors method<sup>2</sup>. Roughly speaking, the kernel method relies primarily on proximity to the query point for selecting $X_{m}$'s, while the nearest neighbors method focuses on the rank of proximity. Notably, discretizing the feature space (also known as quantization), a straightforward yet often effective strategy, can be seen as a variant of the kernel method with static query points and flat kernels.

The problem of estimating conditional distributions can be addressed within the non-parametric regression framework, by employing clustering or resorting to non-parametric least squares, among others. Alternatively, it is feasible to estimate the conditional density function directly: a widely-used method involves estimating the joint and marginal density functions using kernel smoothing and then calculating their ratio. This method shares similarities with the clustering heuristics mentioned earlier. For a more detailed review of these approaches, we refer to Section 1.2.

This work draws inspiration from recent advancements in estimating discrete-time stochastic processes using conditional density function estimation (Pflug and Pichler (2016)) and quantization methods (Backhoff et al. (2022); Acciaio and Hou (2023)). A notable feature of these works is their use of the Wasserstein distance to calculate local errors: the difference between the true and estimated conditional distributions at a query point $x$. One could average these local errors across different values of $x$'s to gauge the global error. Employing Wasserstein distances naturally frames the study within the context of weak convergence, thereby enabling discussions in a relatively general setting, although this approach may yield somewhat weaker results in terms of the mode of convergence. Moreover, utilizing a specific distance rather than the general notion of weak convergence enables a more tangible analysis of the convergence rates and fluctuations. We would like to point out that the advancements made in Backhoff et al. (2022); Acciaio and Hou (2023), as well as our analysis in this paper, relies on recent developments concerning the Wasserstein convergence rate of empirical measures under i.i.d. sampling from a static distribution (cf. Fournier and Guillin (2015)).

### 1.1 Main contributions

First, we introduce some notations to better illustrate the estimators that we study. Let $\mathbb{X}$ and $\mathbb{Y}$ be multi-dimensional unit cubes, with potentially different dimensions, for feature and target spaces. For any integer $M\geq 1$, any $\mathsf{D} = \{(x_m,y_m)\}_{m = 1}^M\in (\mathbb{X}\times \mathbb{Y})^M$, and any Borel set $A\subset \mathbb{X}$, we define a probability measure on $\mathbb{Y}$ by

$$
\hat {\mu} _ {A} ^ {\mathrm {D}} := \left\{ \begin{array}{l l} \left(\sum_ {m = 1} ^ {M} \mathbb {1} _ {A} \left(x _ {m}\right)\right) ^ {- 1} \sum_ {m = 1} ^ {M} \mathbb {1} _ {A} \left(x _ {m}\right) \delta_ {y _ {m}}, & \sum_ {m = 1} ^ {M} \mathbb {1} _ {A} \left(x _ {m}\right) > 0, \\ \lambda_ {\mathbb {Y}}, & \text {otherwise}, \end{array} \right. \tag {1}
$$

where $\lambda_{\mathbb{Y}}$ is the Lebesgue measure on $\mathbb{Y}$ and, for $y \in \mathbb{Y}$, $\delta_y$ is a Dirac measure with atom at $y$. In general, one could consider weighting $\delta_{y_m}$'s (cf. (Györfi et al., 2002, Section 5), (Biau and Devroye, 2015, Chapter 5)), which may offer additional benefits in specific applications. As such adjustments are unlikely to affect the convergence rate, however, we use uniform weighting for simplicity.

With (random) data $\mathcal{D} = \{(X_m, Y_m)\}_{m=1}^M$, we aim to estimate the conditional distribution of $Y$ given $X$. We view this conditional distribution as a measure-valued function $P: \mathbb{X} \to \mathcal{P}(\mathbb{Y})$ and use a subscript for the input argument and write $P_x$. Consider a clustering scheme<sup>3</sup> given by the map $\mathcal{A}^{\mathsf{D}}: \mathbb{X} \to 2^{\mathbb{X}}$. We investigate estimators of the form $x \mapsto \hat{\mu}_{\mathcal{A}^{\mathcal{D}}(x)}^{\mathcal{D}}$. We use $\widehat{P^A}$ to denote said estimator and suppress $\mathcal{D}$ from the notation for convenience. In later sections, we consider two kinds of maps $\mathcal{A}^{\mathsf{D}}$ (i) a ball with fixed radius centered at $x$, called an $r$-box and (ii) the $k$-nearest neighbors of $x$, called $k$-nearest-neighbor estimator. See Definitions 5 and 9 for more details.

One of our main contribution pertains to analyzing the error

$$
\int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \widehat {P} _ {x} ^ {\mathcal {A}}\right) \nu (\mathrm {d} x), \tag {2}
$$

where $\mathcal{W}$ is the 1-Wasserstein distance (see (Villani, 2008, Particular Case 6.2)) and $\nu \in \mathcal{P}(\mathbb{X})$ is arbitrary and provides versatility to the evaluation criterion. A canonical choice for $\nu$ is the Lebesgue measure on $\mathbb{X}$, denoted by $\lambda_{\mathbb{X}}$. This is particularly relevant in control settings where $\mathbb{X}$ represents the state-action space and accurate approximations across various state and action scenarios are crucial for making informed decisions. The form of error above is also foundational in stochastic process estimation under the adapted Wasserstein distance (cf. (Backhoff et al., 2022, Lemma 3.1)), making the techniques we develop potentially relevant in other contexts. Under the assumption that $P$ is Lipschitz continuous (Assumption 2) and standard assumptions on the data collection process (Assumption 3), we analyze the convergence rate and fluctuation by bounding the following two quantities

$$
\mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \widehat {P} _ {x} ^ {\mathcal {A}}\right) \nu (\mathrm {d} x) \right] \quad \mathrm {and} \quad \mathrm {Var} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \widehat {P} _ {x} ^ {\mathcal {A}}\right) \nu (\mathrm {d} x) \right].
$$

Moreover, by analyzing the above quantities, we gain insights into the optimal choice of the clustering mapping $\mathcal{A}$. For the detail statements of these results, we refer to Theorems 7, 10, 8, and 11. We also refer to Section 2.4 for related comments.

To illustrate another aspect of our contribution, we note by design $x \mapsto \widehat{P}_x^{\mathcal{A}}$ is piece-wise constant. This characteristic introduces limitations. Notably, it renders the analysis of performance at the worst-case $x$ elusive. Contrastingly, by building a Lipschitz-continuous parametric estimator $\tilde{P}^\Theta$ from the raw estimator $\widehat{P}^{\mathcal{A}}$, in Proposition 13 we demonstrate that an upper bound on the aforementioned expectation allows us to derive a worst-case performance guarantee. Guided by Proposition 13, we explore a novel approach of training a neural network for estimation, by using $\widehat{P}^{\mathcal{A}}$ as training data and incorporating suitably imposed Lipschitz continuity. To be comprehensive, we include in Section 1.2 a review of studies on Lipschitz continuity in neural networks.

In Section 3.1, we define $\tilde{P}^{\theta}$ as a neural network that approximates $P$, where $\theta$ represents the network parameters. We train $\tilde{P}^{\theta}$ with the objective:

$$
\underset {\theta} {\arg \min} \sum_ {n = 1} ^ {N} \mathcal {W} \left(\widehat {P} _ {\tilde {X} _ {n}} ^ {\mathcal {A}}, \tilde {P} _ {\tilde {X} _ {n}} ^ {\theta}\right),
$$

where $(\tilde{X}_n)_{n=1}^N$ is a set of randomly selected query points. For implementation purposes, we use the $k$-nearest-neighbor estimator in the place of $\hat{P}^{\mathcal{A}}$ (see Definition 9). To mitigate the computational costs stemming from the nearest neighbors search, we employ the technique of Approximate Nearest Neighbor Search with Random Binary Space Partitioning (ANN-RBSP), as discussed in Section 3.1.1. In Section 3.1.2, we compute $\mathcal{W}$ using the Sinkhorn algorithm, incorporating normalization and enforcing sparsity for improved accuracy. To impose a suitable level of local Lipschitz continuity on $\tilde{P}^\theta$, in Section 3.1.3, we employ a neural network with a specific architecture and train the networks using a tailored procedure. The key component of this architecture is the convex potential layer introduced in Meunier et al. (2022). In contrast to most extant literature that imposes Lipschitz continuity on neural networks, our approach does not utilize specific constraint or regularization of the objective function, but relies on certain self-adjusting mechanism embedded in the training.

In Section 3.2, we evaluate the performance of the trained $\tilde{P}^{\theta}$, denoted by $\tilde{P}^{\Theta}$, using three sets of synthetic data in 1D and 3D spaces. Our findings indicate that $\tilde{P}^{\Theta}$ generally outperforms $\widehat{P}^{\mathcal{A}}$, even though it is initially trained to match $\widehat{P}^{\mathcal{A}}$. This superior performance persists even when comparing $\tilde{P}^{\Theta}$ to different $\widehat{P}^{\mathcal{A}}$ using various clustering parameters, without retraining $\tilde{P}^{\Theta}$. Furthermore, despite using the same training parameters, $\tilde{P}^{\Theta}$ consistently demonstrates the ability to adapt to a satisfactory level of local Lipschitz continuity across all cases. Moreover, in one of the test cases, we consider a kernel that exhibits a jump discontinuity, and we find that $\tilde{P}^{\Theta}$ handles this jump case well despite Lipschitz continuity does not hold.

Lastly, we provide further motivation of our approach by highlighting some potential applications for $\tilde{P}^{\Theta}$. The first application is in model-based policy gradient method in reinforcement learning. We anticipate that the enforced Lipschitz continuity allows us to directly apply the policy gradient update via compositions of $\tilde{P}^{\Theta}$ and cost function for more effective optimality searching. The second application of $\tilde{P}^{\Theta}$ is in addressing optimisation in risk-averse Markov decision processes, where dynamic programming requires knowledge beyond the conditional expectation of the risk-to-go (cf. Chow et al. (2015); Huang and Haskell (2017); Coache et al. (2023); Cheng and Jaimungal (2023)). The study of these applications is left for further research.

### 1.2 Related works

In this section, we will first review the clustering approach in estimating conditional distributions, and then proceed to review recent studies on Lipschitz continuity in neural networks.

#### 1.2.1 ESTIMATING CONDITIONAL DISTRIBUTIONS VIA CLUSTERING

The problem of estimating conditional distributions is frequently framed as non-parametric regression problems for real-valued functions. For instance, when $d_{\mathbb{Y}} = 1$, estimate the conditional $\alpha$-quantile of $Y$ given $X$. Therefore, we begin by reviewing some of the works in non-parametric regression.

The kernel method in non-parametric regression traces its origins back to the Nadaraya-Watson estimator (Nadaraya (1964); Watson (1964)), if not earlier. Subsequent improvements have been introduced, such as integral smoothing (Gasser and Müller (1979), also known as the Gasser-Müller estimator), local fitting with polynomials instead of constants (Fan (1992)), and adaptive kernels (Hall et al. (1999)). Another significant area of discussion is the choice of kernel bandwidth, as detailed in works like Hardle and Marron (1985); Gijbels and Goderniaux (2004); Kohler et al. (2014). Regarding convergence rates, analyses under various settings can be found in Stone (1982); Hall and Hart (1990); Kohler et al. (2009, 2010), with Stone (1982) being particularly relevant to our study for comparative purposes. According to Stone (1982), if the target function is Lipschitz continuous, with i.i.d. sampling and that the sampling distribution in the feature space has a uniformly positive density, then the optimal rate of the $\| \cdot \| _1$-distance between the regression function and the estimator is of the order $M^{-\frac{1}{d_{\mathbb{X}} + 2}}$. For a more comprehensive review of non-parametric regression using kernel methods, we refer to the books such as Györfi et al. (2002); Ferraty and Vieu (2006); Wasserman (2006) and references therein.

Non-parametric regression using nearest neighbors methods originated from classification problems (Fix and Hodges (1951)). Early developments in this field can be found in Mack (1981); Devroye (1982); Bhattacharya and Gangopadhyay (1990). For a comprehensive introduction to nearest neighbors methods, we refer to Györfi et al. (2002). More recent reference such as Biau and Devroye (2015) offers further detailed exploration of the topic. The nearest neighbor method can be viewed as a variant of the kernel method that adjusts the bandwidth based on the number of local data points—a property that has gained significant traction. Recently, the application of the nearest neighbor method has expanded into various less standard settings, including handling missing data (Rachdi et al. (2021)), reinforcement learning (Shah and Xie (2010); Giegrich et al. (2024)), and time series forecasting (Martínez et al. (2017)). For recent advancements in convergence analysis beyond the classical setting, see Zhao and Lai (2019); Padilla et al. (2020); Ryu and Kim (2022); Demirkayaa et al. (2024).

Although the review above mostly focuses on clustering approach, other effective approaches exist, such as non-parametric least square, or more broadly, conditional elicitability (e.g., Györfi et al. (2002); Muandet et al. (2017); Wainwright (2019); Coache et al. (2023)). Non-parametric least square directly fits the data using a restricted class of functions. At first glance, this approach appears distinct from clustering. However, they share some similarities in their heuristics: the rigidity of the fitting function, due to imposed restrictions, allows data points near the query point to affect the estimation, thereby implicitly incorporating elements of clustering.

Apart from non-parametric regression, conditional density function estimation is another significant method for estimating conditional distributions. One approach is based on estimating joint and marginal density functions, and then using the ratio of these two to produce an estimator for the conditional density function. A key technique used in this approach is kernel smoothing. Employing a static kernel for smoothing results in a conditional density estimator that shares similar clustering heuristics to those found in the kernel method of non-parametric regression. For a comprehensive overview of conditional density estimation, we refer to reference books such as Scott (2015); Simonoff (1996). For completeness, we also refer to (Muandet et al., 2017, Section 5.1) for a perspective on static density function estimation from the standpoint of reproducing kernel Hilbert space. Further discussions on estimation using adaptive kernels can be found in, for example, Bashtannyk and Hyndman (2001); Lacour (2007); Bertin et al. (2016); Zhao and Tabak (2023).

Despite extensive research in non-parametric regression and conditional density function estimation, investigations from the perspective of weak convergence have been relatively limited, only gaining more traction in the past decade. Below, we highlight a few recent studies conducted in the context of estimating discrete-time stochastic processes under adapted Wasserstein distance, as the essence of these studies are relevant to our evaluation criterion (2). Pflug and Pichler (2016) explores the problem asymptotically, employing tools from conditional density function estimation with kernel smoothing. Subsequently, Backhoff et al. (2022) investigates a similar problem with a hypercube as state space, employing the quantization method. Their approach removes the need to work with density functions. They calculate the convergence rate, by leveraging recent developments in the Wasserstein convergence rate of empirical measures Fournier and Guillin (2015). Moreover, a sub-Gaussian concentration with parameter $M^{-1}$ is established. The aforementioned results are later extended to $\mathbb{R}^d$ in Acciaio and Hou (2023), where a non-uniform grid is used to mitigate assumptions on moment conditions. Most recently, Hou (2024) examines smoothed variations of the estimators proposed in Backhoff et al. (2022); Acciaio and Hou (2023). Other developments on estimators constructed from smoothed quantization can be found in Smid and Kozmik (2024).

Lastly, regarding the machine learning techniques used in estimating conditional distributions, conditional generative models are particularly relevant. For reference, see Mirza and Osindero (2014); Papamakarios et al. (2017); Vaswani et al. (2017); Fetaya et al. (2020). These models have achieved numerous successes in image generation and natural language processing. We suspect that, due to the relatively discrete (albeit massive) feature spaces in these applications, clustering is implicitly integrated into the training procedure. In continuous spaces, under suitable setting, clustering may also become an embedded part of the training procedure. For example, implementations in Li et al. (2020b); Vuletic et al. (2024); Hosseini et al. (2024) do not explicitly involve clustering and use training objectives that do not specifically address the issues highlighted in the motivating example at the beginning of the introduction. Their effectiveness could possibly be attributed to certain regularization embedded within the neural network and training procedures. Nevertheless, research done in continuous spaces that explicitly uses clustering approaches when training conditional generative models holds merit. Such works are relatively scarce. For an example of this limited body of research, we refer to Xu and Acciaio (2022), where the conditional density function estimator from Pflug and Pichler (2016) is used to train an adversarial generative network for stochastic process generation.

#### 1.2.2 LIPSCHITZ CONTINUITY IN NEURAL NETWORKS

Recently, there has been increasing interest in understanding and enforcing Lipschitz continuity in neural networks. The primary motivation is to provide a certifiable guarantee for classification tasks performed by neural networks: it is crucial that minor perturbations in the input object have a limited impact on the classification outcome.

One strategy involves bounding the Lipschitz constant of a neural network, which can then be incorporated into the training process. For refined upper bounds on the (global) Lipschitz constant, see, for example, Bartlett et al. (2017); Virmaux and Scaman (2018); Tsuzuki et al. (2018); Fazlyab et al. (2019); Xue et al. (2022); Fazlyab et al. (2024). For local bounds, we refer to Jordan and Dimakis (2021); Bhowmick et al. (2021); Shi et al. (2022) and the references therein. We also refer to Zhang et al. (2022) for a study of the Lipschitz property from the viewpoint of boolean functions.

Alternatively, designing neural network architectures that inherently ensure desirable Lipschitz constants is another viable strategy. Works in this direction include Meunier et al. (2022); Singla et al. (2022); Wang and Manchester (2023); Araujo et al. (2023). Notably, the layer introduced in Meunier et al. (2022) belongs to the category of residual connection (He et al. (2016)).

Below, we review several approaches that enforce Lipschitz constants during neural network training. Tsuzuku et al. (2018); Liu et al. (2022) explore training with a regularized objective function that includes upper bounds on the network's Lipschitz constant. Gouk et al. (2021) frame the training problem into constrained optimization and train with projected gradients descent. Given the specific structure of the refined bound established in Fazlyab et al. (2019), Pauli et al. (2022) combines training with semi-definite programming. They develop a version with a regularized objective function and another that enforces the Lipschitz constant exactly. Fazlyab et al. (2024) also investigates training with a regularized objective but considers Lipschitz constants along certain directions. Huang et al. (2021) devises a training procedure that removes components from the weight matrices to achieve smaller local Lipschitz constants. Trockman and Kolter (2021) initially imposes orthogonality on the weight matrices, and subsequently enforces a desirable Lipschitz constant based on that orthogonality. Ensuring desirable Lipschitz constants with tailored architectures, Singla et al. (2022); Wang and Manchester (2023) train the networks directly. Although the architecture proposed in Meunier et al. (2022) theoretically ensures the Lipschitz constant, it requires knowledge of the spectral norm of the weight matrices, which does not admit explicit expression in general. Their training approach combines power iteration for spectral norm approximation with the regularization methods used in Tsuzuku et al. (2018).

Finally, we note that due to their specific application scenarios, these implementations concern relatively stringent robustness requirements and thus necessitate more specific regularization or constraints. In our setting, it is generally desirable for the neural network to automatically adapt to a suitable level of Lipschitz continuity based on the data, while also avoiding excessive oscillations from over-fitting. The literature directly addressing this perspective is limited (especially in the setting of conditional distribution estimation). We refer to Bai et al. (2021); Bountakas et al. (2023); Cohen et al. (2019) for discussions that could be relevant.

### 1.3 Organization of the paper

Our main theoretical results are presented in Section 2. Section 3 is dedicated to the training of $\tilde{P}^{\Theta}$. We will outline the key components of our training algorithm and demonstrate its performance on three sets of synthetic data. We will prove the theoretical results in Section 4. Further implementation details and ablation analysis are provided in Section 5. In Section 6, we discuss the weaknesses and potential improvements of our implementation. Appendix B and C respectively contain additional plots and a table that summarizes the configuration of our implementation. Additionally, Appendix D includes a rougher version of the fluctuation results.

## Notations and terminologies

Throughout, we adopt the following set of notations and terminologies.

- On any normed space $(E, \| \cdot \|)$, $B(x,\gamma)$ denotes the closed ball of radius $\gamma$ around $x$, namely $B(x,\gamma) = \{x' \in E \mid \| x - x' \| \leq \gamma\}$.
- For any measurable space $(E,\mathcal{E})$, $\mathcal{P}(E)$ denotes the set of probability distributions on $(E,\mathcal{E})$. For all $x\in E$, $\delta_x\in \mathcal{P}(E)$ denotes the Dirac mass at $x$.
- We endow normed spaces $(E, \| \cdot \|)$ with their Borel sigma-algebra $\mathcal{B}(E)$, and $\mathcal{W}$ denotes the 1-Wasserstein distance on $\mathcal{P}(E)$.
- On $\mathbb{X} = [0,1]^d$, we denote by $\lambda_{\mathbb{X}}$ the Lebesgue measure. We say a measure $\nu \in \mathcal{P}(\mathbb{X})$ is dominated by Lebesgue measure with a constant $\overline{C} > 0$ if $\nu(A) \leq \overline{C}\lambda_{\mathbb{X}}(A)$ for all $A \in \mathcal{B}([0,1]^d)$.
- The symbol $\sim$ denotes equivalence in the sense of big O notation, indicating that each side dominates the other up to a multiplication of some positive absolute constant. More precisely, $a_{n} \sim b_{n}$ means there are finite constants $c, C > 0$ such that

$$
c a _ {n} \leq b _ {n} \leq C a _ {n}, \quad n \in \mathbb {N}.
$$

Similarly, $\lesssim$ implies that one side is of a lesser or equal, in the sense of big O notation, compared to the other.

## 2. Theoretical results

In Section 2.1, we first formally set up the problem and introduce some technical assumption. We then study in Section 2.2 and 2.3 the convergence and fluctuation of two versions of $\hat{P}^{\mathcal{A}}$, namely, the $r$-box estimator and the $k$-nearest-neighbor estimator. Related comments are organized in Section 2.4. Moreover, in Section 2.5, we provide a theoretical motivation for the use of $\tilde{P}^{\Theta}$, the Lipschitz-continuous parametric estimator trained from $\hat{P}^{\mathcal{A}}$.

### 2.1 Setup

For $d_{\mathbb{X}}, d_{\mathbb{Y}} \geq 1$ two integers, we consider $\mathbb{X} := [0,1]^{d_{\mathbb{X}}}$ and $\mathbb{Y} := [0,1]^{d_{\mathbb{Y}}}$, endowed with their respective sup-norm $\| \cdot \|_{\infty}$.

**Remark 1** The sup-norm is chosen for simplicity of the theoretical analysis only: as all norms on $\mathbb{R}^n$ are equivalent (for any generic $n\geq 1$), our results are valid, up to different multiplicative constants, for any other choice of norm.

We aim to estimate an unknown probabilistic kernel

$$
\begin{array}{l} P: \mathbb {X} \to \mathcal {P} (\mathbb {Y}) \\ x \mapsto P _ {x} (\mathrm {d} y). \\ \end{array}
$$

To this end, given an integer-valued sample size $M \geq 1$, we consider a set of (random) data points $\mathcal{D} := \{(X_m, Y_m)\}_{m=1}^M$ associated to $P$. We also define the set of projections of the data points onto the feature space as $\mathcal{D}_{\mathbb{X}} := \{X_m\}_{m=1}^M$.

Throughout this section, we work under the following technical assumptions.

**Assumption 2** (Lipschitz continuity of kernel) There exists $L \geq 0$ such that, for all $(x, x') \in \mathbb{X}^2$,

$$
\mathcal {W} \left(P _ {x}, P _ {x ^ {\prime}}\right) \leq L \| x - x ^ {\prime} \| _ {\infty}.
$$

**Assumption 3** The following is true:

(i) $\mathcal{D}$ is i.i.d. with probability distribution $\psi \coloneqq \xi \otimes P$, where $\xi \in \mathcal{P}(\mathbb{X})$ and where $\xi \otimes P \in \mathcal{P}(\mathbb{X} \times \mathbb{Y})$ is (uniquely, by Caratheodory extension theorem) defined by

$$
(\xi \otimes P) (A \times B) := \int_ {\mathbb {X}} \mathbb {1} _ {A} (x) P _ {x} (B) \xi (\mathrm {d} x), \quad A \in \mathcal {B} (\mathbb {X}), B \in \mathcal {B} (\mathbb {Y}).
$$

(ii) There exists $\underline{c} \in (0,1]$ such that, for all $A \in \mathcal{B}(\mathbb{X})$, $\xi(A) \geq \underline{c} \lambda_{\mathbb{X}}(A)$.

These assumptions allow us to analyze convergence and gain insights into the optimal clustering hyper-parameters without delving into excessive technical details. Assumption 2 is mainly used for determining the convergence rate. If the convergence rate is not of concern, it is possible to establish asymptotic results with less assumptions. We refer to Devroye (1982); Backhoff et al. (2022) for relevant results. The conditions placed on $\xi$ in Assumption 3 are fairly standard, though less stringent alternatives are available. For instance, Assumption 3 (i) can be weakened by considering suitable dependence Hall and Hart (1990) or ergodicity in the context of stochastic processes Rudolf and Schweizer (2018). Assumption 3 (ii), implies there is mass almost everywhere and is aligned with the motivation from control settings discussed in the introduction. Assumptions 2 and 3 are not exceedingly stringent and provides a number of insights into the estimation problem. More general settings are left for further research.

The estimators discussed in subsequent sections are of the form $\widehat{P^{\mathcal{A}}}$, as introduced right after (1), for two specific choices of clustering schemes $\mathcal{A}$ constructed with the data $\mathcal{D}$.

**Remark 4** In the following study, we assert all the measurability needed for $\widehat{P^A}$ to be well-defined. These measurability can be verified using standard measure-theoretic tools listed in, for example, (Aliprantis and Border, 2006, Section 4 and 15).

### 2.2 Results on $r$-box estimator

The first estimator, which we term the $r$-box estimator, is defined as follows.

**Definition 5** Choose $r$, a real number, s.t. $0 < r < \frac{1}{2}$. The $r$-box estimator for $P$ is defined by

$$
\begin{array}{l} \hat {P} ^ {r}: \mathbb {X} \rightarrow \mathcal {P} (\mathbb {Y}) \\ x \mapsto \hat {P} _ {x} ^ {r} := \hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D}}, \\ \end{array}
$$

where, for all $x \in \mathbb{X}$, $\mathcal{B}^r(x) \coloneqq B(\beta^r(x), r)$ and $\beta^r(x) \coloneqq r \vee x \wedge (1 - r)$, where $r \vee \cdot$ and $\cdot \wedge (1 - r)$ are applied entry-wise.

**Remark 6** The set $\mathcal{B}^r (x)$ is defined such that it is a ball of radius around $x$ whenever $x$ is at least $r$ away from the boundary $\partial \mathbb{X}$ (in all of its components), otherwise, we move the point $x$ in whichever components are within $r$ from $\partial \mathbb{X}$ to be a distance $r$ away from $\partial \mathbb{X}$. Consequently, for all $0 < r < \frac{1}{2}$ and for all $x\in \mathbb{X}$, $\mathcal{B}^r (x)$ has a bona fide radius of $r$, as the center $\beta^r (x)$ is smaller or equal to $r$ away from $\partial \mathbb{X}$.

For the $r$-box estimator, we have the following convergence results. The theorem below discusses the convergence rate of the average Wasserstein distance between the unknown kernel evaluated at any point and its estimator, when the radius $r$ is chosen optimally with respect to the data sample $M$. Section 4.2 is dedicated to its proof.

**Theorem 7** Under Assumptions 2 and 3, choose $r$ as follows

$$
r \sim \left\{ \begin{array}{l l} M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}}, & d _ {\mathbb {Y}} = 1, 2 \\ M ^ {- \frac {1}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}}, & d _ {\mathbb {Y}} \geq 3. \end{array} \right.
$$

Then, there is a constant $C > 0$ (which depends only on $d_{\mathbb{X}}, d_{\mathbb{Y}}, L, \underline{c}$), such that, for all probability distribution $\nu \in \mathcal{P}(\mathbb{X})$, we have

$$
\mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \hat {P} _ {x} ^ {r}\right) \nu (\mathrm {d} x) \right] \leq \sup  _ {x \in \mathbb {X}} \mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \hat {P} ^ {r}\right) \right] \leq C \times \left\{ \begin{array}{l l} M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}}, & d _ {\mathbb {Y}} = 1, \\ M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}} \ln (M), & d _ {\mathbb {Y}} = 2, \\ M ^ {- \frac {1}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}}, & d _ {\mathbb {Y}} \geq 3. \end{array} \right. \tag {3}
$$

Next, we bound the associated variance whose proof is postponed to Section 4.3.

**Theorem 8** Under Assumptions 3, consider $r \in (0, \frac{1}{2}]$. Let $\nu \in \mathcal{P}(\mathbb{X})$ be dominated by $\lambda_{\mathbb{X}}$ with a constant $\overline{C} > 0$. Then,

$$
\mathrm {Var} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \hat {P} _ {x} ^ {r}\right) \mathrm {d} \nu (x) \right] \leq \frac {4 ^ {d _ {\mathbb {X}} + 1} \overline {{C}} ^ {2}}{\underline {{c}} ^ {2} (M + 1)}.
$$

### 2.3 Results on $k$-nearest-neighbor estimator

Here, we focus in the second estimator - the $k$-nearest-neighbor estimator, defined as follows.

**Definition 9** Let $k \geq 1$ an integer. The $k$-nearest-neighbor estimator for $P$ is defined by

$$
\check {P} ^ {k}: \mathbb {X} \rightarrow \mathcal {P} (\mathbb {Y})
$$

$$
x \mapsto \check {P} _ {x} ^ {k} := \hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}} (x)}} ^ {\mathcal {D}},
$$

where, for any integer $M \geq 1$ and any $\mathsf{D}_{\mathbb{X}} \in \mathbb{X}^M$, $\mathcal{N}^{k,\mathsf{D}_{\mathbb{X}}}(\boldsymbol{x})$ contains (exactly) $k$ points of $\mathsf{D}_{\mathbb{X}}$ which are closest to $x$, namely

$$
\mathcal {N} ^ {k, \mathrm {D} _ {\mathbb {X}}} (x) := \left\{x ^ {\prime} \in \mathrm {D} _ {\mathbb {X}} \mid \| x - x ^ {\prime} \| _ {\infty} \text{ is among the } k \text{-smallest of } \left(\| x - x ^ {\prime} \| _ {\infty}\right) _ {x ^ {\prime} \in \mathrm {D} _ {\mathbb {X}}} \right\},
$$

Here, in case of a tie when choosing the $k$-th smallest, we break the tie randomly with uniform probability.

We have the following analogs of the convergence results (Theorems 7 and 8) for the $k$-nearest-neighbor estimator. The proofs are postponed to Section 4.4 and Section 4.5, respectively.

**Theorem 10** Under Assumptions 2 and 3, and choosing $k$ as

$$
k \sim \left\{ \begin{array}{l l} M ^ {\frac {2}{d _ {\mathbb {X}} + 2}}, & d _ {\mathbb {Y}} = 1, 2, \\ M ^ {\frac {d _ {\mathbb {Y}}}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}}, & d _ {\mathbb {Y}} \geq 3, \end{array} \right.
$$

there is a constant $C > 0$ (which depends only on $d_{\mathbb{X}}, d_{\mathbb{Y}}, L, \underline{c}$), such that, for all probability distribution $\nu \in \mathcal{P}(\mathbb{X})$, we have

$$
\mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \nu (\mathrm {d} x) \right] \leq \sup  _ {x \in \mathbb {X}} \mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \right] \leq C \times \left\{ \begin{array}{l l} M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}}, & d _ {\mathbb {Y}} = 1, \\ M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}} \ln M, & d _ {\mathbb {Y}} = 2, \\ M ^ {- \frac {1}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}}, & d _ {\mathbb {Y}} \geq 3. \end{array} \right. \tag {4}
$$

**Theorem 11** Under Assumptions 3, for any $\nu \in \mathcal{P}(\mathbb{X})$, we have

$$
\operatorname {Var} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \nu (\mathrm {d} x) \right] \leq \frac {1}{k}. \tag {5}
$$

Moreover, if $\nu$ is dominated by $\lambda_{\mathbb{X}}$ with a constant $\overline{C} > 0$, then

$$
\begin{array}{l} \mathrm {Var} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \nu (\mathrm {d} x) \right] \\ \leq \frac {2 ^ {2 d _ {\mathbb {X}} + 1} \overline {{C}} ^ {2} M}{\underline {{c}} ^ {2} k ^ {2}} \left(\left(8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} + \frac {k}{M - 1}\right) ^ {2} + \frac {\sqrt {2 \pi}}{\sqrt {M - 1}} \left(8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} + \frac {k}{M - 1}\right) + \frac {4}{M - 1}\right). \\ \end{array}
$$

With $k$ chosen as in Theorem 10, this reduces to

$$
\operatorname {Var} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \nu (\mathrm {d} x) \right] \lesssim \left\{ \begin{array}{l l} M ^ {- \frac {2 (2 \vee d _ {\mathbb {Y}})}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}} \ln (M), & 2 \vee d _ {\mathbb {Y}} \leq d _ {\mathbb {X}}, \\ M ^ {- 1}, & 2 \vee d _ {\mathbb {Y}} > d _ {\mathbb {X}}. \end{array} \right.
$$

### 2.4 Comments on the convergence rate

This section gathers several comments on the convergence results we have developed in Sections 2.2 and 2.3.

#### 2.4.1 ON THE CONVERGENCE RATE

We first comment on the expectations in Theorem 7 and 10.

**Sharpness of the bounds.** Currently, we cannot establish the sharpness of the convergence rates in Theorems 7 and 10. However, we can compare our results to established results in similar settings. For $d_{\mathbb{Y}} = 1$, we may compare it to the optimal rate of non-parametric regression of a Lipschitz continuous function. It is shown in Stone (1982) that the optimal rate is $M^{-\frac{1}{d_{\mathbb{X}} + 2}}$, the same as in Theorems 7 and 10 when $d_{\mathbb{Y}} = 1$. For $d_{\mathbb{Y}} \geq 3$, as noted in Backhoff et al. (2022), we may compare to the Wasserstein convergence rate of empirical measure in the estimation of a static distribution on $\mathbb{R}^{d_{\mathbb{X}} + d_{\mathbb{Y}}}$. We refer to Fournier and Guillin (2015) for the optimal rate, which coincides with those in Theorems 7 and 10.

**Error components.** We discuss the composition of our upper bound on the expected average Wasserstein error by dissecting the proof of Theorem 7 and 10. In the proofs, we decompose the expected average errors into two components: approximation error and estimation error. The approximation error occurs when treating $P_{x'}$ as equal to $P_x$ when $x'$ is close to the query point $x$, leading to an error of size $L\|x - x'\|_{\infty}$. The estimation error is associated with the Wasserstein error of empirical measure under i.i.d. sampling (see (21)). From Definitions 5 and 9, the $r$-box estimator effectively manages the approximation error but struggles with controlling the estimation error, whereas the $k$-nearest-neighbor estimator exhibits the opposite behavior.

**Explicit bounds.** We primarily focus on analyzing the convergence rates of the $r$-box and $k$-nearest-neighbor estimators as $M \to \infty$. Therefore, within the proofs of these results, we track only the rates (and ignore various constant coefficients). If more explicit bounds are preferred, intermediate results such as (23), or (27) could be good starting points for computing them. Additionally, in Section A, we provide a numerical illustration that highlights the impact of the Lipschitz constant $L$ and the choices of $r$ and $k$ on the bounds.

#### 2.4.2 ON THE FLUCTUATION

We next discuss the variances studied in Theorems 8 and 11. In Appendix D, we also include results derived from the Azuma-Hoeffding inequality (e.g., (Wainwright, 2019, Corollary 2.20)), though they provide rougher rates.

**Condition that $\nu$ is dominated by $\lambda_{\mathbb{X}}$.** In Theorems 8 and 11, we assume that the $\nu$ is dominated by $\lambda_{\mathbb{X}}$. This assumption is somewhat necessary. To illustrate, let us examine the nonparametric regression problem under a comparable scenario. We consider a fixed query point. In this context, the central limit theorem for $k$-nearest-neighbor estimator is well-established, and the normalizing rate is $k^{-\frac{1}{2}}$ (cf. (Biau and Devroye, 2015, Theorem 14.2)). This suggests that the rate in (5) is sharp. For the $r$-box estimator, we believe that a supporting example can be constructed where $\nu$ is highly concentrated. On the other hand, we conjecture that if $\xi \sim \nu$, the variance could potentially attain the order of $M^{-1}$. For a pertinent result, we direct the reader to (Backhoff et al., 2022, Theorem 1.7).

**Sharpness of the bounds.** Regarding the variance in Theorem 8, it is upper bounded by the commonly observed order of $M^{-1}$. We believe that this rate is sharp, though we do not have a proof at this time. As for Theorem 11, the variance is subject to a rougher rate when $2 \vee d_{\mathbb{Y}} \leq d_{\mathbb{X}}$. We, however, conjecture that this variance attains the order of $M^{-1}$ as long as $\nu$ is dominated by $\lambda_{\mathbb{X}}$.

### 2.5 Towards implementation with neural networks

In light of recent practices in machine learning, during the learning of $P$, we may combine the $r$-box method or $k$-nearest-neighbor method into the training of certain parameterized model. To this end we let

$$
\tilde {P}: \mathbb {T} \times \mathbb {X} \to \mathcal {P} (\mathbb {Y})
$$

$$
(\theta , x) \mapsto \tilde {P} _ {x} ^ {\theta}
$$

be a parameterized model (e.g., a neural network), where $\mathbb{T}$ is the parameter space and $\theta \in \mathbb{T}$ is the parameter to be optimized over. Given an integer $N \geq 1$, we may train $\tilde{P}^{\theta}$ on a set of query points $\mathcal{Q} = (\tilde{X}_n)_{n=1}^N$ satisfying the assumption below.

**Assumption 12** The query points $\mathcal{Q} = \{(\tilde{X}_n)\}_{n=1}^N$ are i.i.d. with uniform distribution over $\mathbb{X}$, and are independent of the data points $\mathcal{D} = \{(X_m, Y_m)\}_{m=1}^M$.

We propose the training objectives below

$$
\underset {\theta \in \mathbb {T}} {\arg \min } \frac {1}{N} \sum_ {n = 1} ^ {N} \mathcal {W} \left(\hat {P} _ {\tilde {X} _ {n}} ^ {r}, \tilde {P} _ {\tilde {X} _ {n}} ^ {\theta}\right) \quad \text {or} \quad \underset {\theta \in \mathbb {T}} {\arg \min } \frac {1}{N} \sum_ {n = 1} ^ {N} \mathcal {W} \left(\check {P} _ {\tilde {X} _ {n}} ^ {k}, \tilde {P} _ {\tilde {X} _ {n}} ^ {\theta}\right), \tag {6}
$$

that is, minimize the mean of 1-Wasserstein errors between the parametrized model and the empirical $r$-box (or $k$-nearest-neighbour) approximation of the conditional distribution at the location of the random query points.

The following proposition together with Theorem 7 or Theorem 10 justifies using the objectives in (6). It is valid for any estimator for $P$ that satisfies the bounds in (3) or (4). Centered on appropriate Lipschitz continuity conditions, the proposition offers insights into the worst-case performance guarantees. The proof is deferred to Section 4.6. We also refer to Altekrüger et al. (2023) for a worst-case performance guarantee for conditional generative models, which is contingent upon Lipschitz continuity. For related practical approaches, we refer to, for example, Nguyen et al. (2024) and the references therein. In contrast, similar guarantees for the $r$-box and $k$-nearest-neighbor estimators are more elusive due to their inherently piece-wise constant nature.

**Proposition 13** Suppose Assumptions 2, 3, and 12 hold. Let $\overline{P}$ of $P$ be an estimator constructed using the data points $\mathcal{D}$ only. Consider a training procedure that produces a (random) $\Theta = \Theta(\mathcal{D}, \mathcal{Q})$ satisfying

$$
\sup  _ {x, x ^ {\prime} \in \mathbb {X}} \frac {\mathcal {W} \left(\tilde {P} _ {x} ^ {\Theta} , \tilde {P} _ {x ^ {\prime}} ^ {\Theta}\right)}{\| x - x ^ {\prime} \| _ {\infty}} \leq L ^ {\Theta} \tag {7}
$$

for some (random) $L^{\Theta} > 0$. Then,

$$
\begin{array}{l} \mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \tilde {P} _ {x} ^ {\Theta}\right) \mathrm {d} x \right] \leq \mathbb {E} \left[ (L + L ^ {\Theta}) \mathcal {W} \left(\lambda_ {\mathbb {X}}, \frac {1}{N} \sum_ {n = 1} ^ {N} \delta_ {\tilde {X} _ {n}}\right) \right] \tag {8} \\ + \mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} (P _ {x}, \overline {{P}} _ {x}) \mathrm {d} x \right] + \mathbb {E} \left[ \frac {1}{N} \sum_ {n = 1} ^ {N} \mathcal {W} (\overline {{P}} _ {\tilde {X} _ {n}}, \tilde {P} _ {\tilde {X} _ {n}} ^ {\Theta}) \right]. \\ \end{array}
$$

Moreover, with probability 1,

$$
\sup  _ {x \in \mathbb {X}} \mathcal {W} \left(P _ {x}, \tilde {P} _ {x} ^ {\Theta}\right) \leq \left(d _ {\mathbb {X}} + 1\right) ^ {\frac {1}{d _ {\mathbb {X}} + 1}} \left(L + L ^ {\Theta}\right) ^ {\frac {d _ {\mathbb {X}}}{d _ {\mathbb {X}} + 1}} \left(\int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \tilde {P} _ {x} ^ {\Theta}\right) \mathrm {d} x\right) ^ {\frac {1}{d _ {\mathbb {X}} + 1}}. \tag {9}
$$

**Remark 14** Assuming $L^{\Theta} \leq \overline{L}$ for some (deterministic) $\overline{L} > 0$, by (9) and Jensen's inequality, we have

$$
\mathbb {E} \left[ \sup _ {x \in \mathbb {X}} \mathcal {W} \left(P _ {x}, \tilde {P} _ {x} ^ {\Theta}\right) \right] \leq (d _ {\mathbb {X}} + 1) ^ {\frac {1}{d _ {\mathbb {X}} + 1}} (L + \overline {{L}}) ^ {\frac {d _ {\mathbb {X}}}{d _ {\mathbb {X}} + 1}} \mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} (P _ {x}, \tilde {P} _ {x} ^ {\Theta}) \mathrm {d} x \right] ^ {\frac {1}{d _ {\mathbb {X}} + 1}}.
$$

This together with (8) provides a worst-case performance guarantee for $\tilde{P}^{\Theta}$.

**Remark 15** Proposition 13 along with Remark 14 provides insights into the worst-case performance guarantees, but more analysis is needed. Specifically, understanding the magnitude of $L^{\Theta}$ and $\mathbb{E}\left[\frac{1}{N}\sum_{n = 1}^{N}\mathcal{W}(\overline{P}_{\tilde{X}_n},\tilde{P}_{\tilde{X}_n}^{\Theta})\right]$ requires deeper knowledge of the training processes for $\tilde{P}^{\Theta}$, which are currently not well understood in the extant literature. Alternatively, in the hypothetical case where $\tilde{P}^{\Theta} = P$, $L^{\Theta}$ would match $L$ as specified in Assumption 2, and $\mathbb{E}\left[\frac{1}{N}\sum_{n = 1}^{N}\mathcal{W}\left(\overline{P}_{\tilde{X}_n},\tilde{P}_{\tilde{X}_n}^{\Theta}\right)\right]$ would obey Theorem 7 or 10. However, practical applications must also consider the universal approximation capability of $\tilde{P}^{\theta}$. To the best of our knowledge, research on universal approximation with regularity constraints remains relatively limited. For a somewhat related study, we refer to Hong and Kratsios (2024) who explore the approximation of real-valued functions under Lipschitz continuity constraints.

## 3. Implementation with neural networks

Let $\mathbb{X}$ and $\mathbb{Y}$ be equipped with $\| \cdot \| _1$. Following the discussion in Section 2.5, we let $\tilde{P}^{\theta}:\mathbb{X}\to \mathcal{P}(\mathbb{Y})$ be parameterized by a neural network and develop an algorithm that trains $\tilde{P}^{\theta}$ based on $k$-nearest-neighbor estimator. The $k$-nearest-neighbor estimator $\tilde{P}^{k}$ is preferred as $\check{P}_x^k$ consistently outputs $k$ atoms. This regularity greatly facilitates implementation. For instance, it enables the use of 3D tensors during Sinkhorn iterations to enhance execution speed (see Section 3.1.2 later). We refer also to the sparsity part of Section 5.2 for another component that necessitates the aforementioned regularity of $\tilde{P}^k$. These components would not be feasible with the $r$-box estimator $\hat{P}^r$, as $\hat{P}_x^r$ produces an undetermined number of atoms. Furthermore, there is a concern that in some realizations, $\hat{P}_x^r$ at certain $x$ may contain too few data points, potentially leading $\tilde{P}_x^\Theta$ to exhibit unrealistic concentration.

We next provide some motivation for this implementation. For clarity, we refer to the $r$-box estimator and the $k$-nearest-neighbor estimator as raw estimators. Additionally, we refer to $\tilde{P}^{\Theta}$, once trained, as the neural estimator. While raw estimators are adequate for estimating $P$ on their own, they are piece-wise constant in $x$ by design. On the other hand, a neural estimator is continuous in $x$. This continuity provides a performance guarantee in $\sup \mathcal{W}$ distance, as outlined in Proposition 13 and the following remark. Moreover, the neural estimator inherently possesses gradient information. As discussed in the introduction, this feature renders the neural estimators useful in downstream contexts where gradient information is important, e.g., when performing model-based reinforcement learning.

We construct $\tilde{P}^{\theta}$ such that it maps $x\in \mathbb{X}$ to atoms in $\mathbb{Y}$ with equal probabilities. For the related universal approximation theorems, we refer to Kratsios (2023); Acciaio et al. (2024). We represent these atoms with a vector with $N_{\mathrm{atom}}$ entries denoted by $y^\theta (x) = (y_1^\theta (x),\ldots ,y_{N_{\mathrm{atom}}}^\theta (x))\in \mathbb{Y}^{N_{\mathrm{atom}}}$ where $N_{\mathrm{atom}}\in \mathbb{N}$ is chosen by the user. In our implementation, we set $N_{\mathrm{atom}} = k$. To be precise, we construct $\tilde{P}^{\theta}$ such that

$$
\tilde {P} _ {x} ^ {\theta} = \frac {1}{N _ {\mathrm {a t o m}}} \sum_ {j = 1} ^ {N _ {\mathrm {a t o m}}} \delta_ {y _ {j} ^ {\theta} (x)}, \quad x \in \mathbb {N}. \tag {10}
$$

This is known as the Lagrangian discretization (see (Peyre and Cuturi, 2019, Section 9)). In Algorithm 1, we present a high level description of our implementation of training $\tilde{P}^{\theta}$ based on the raw $k$-nearest-neighbor estimator.

**Algorithm 1** Deep learning conditional distribution in conjunction with $k$-NN estimator  
**Input**: data $\{(X_m,Y_m)\}_{m = 1}^M$ valued in $\mathbb{R}^{d_{\mathbb{X}}}\times \mathbb{R}^{d_{\mathbb{Y}}}$, neural estimator $\tilde{P}^{\theta}$ represented by $y^\theta (x)$ as elaborated in (10), parameters such as $k,N_{\mathrm{atoms}},N_{\mathrm{batch}}\in \mathbb{N}_{+}$, and learning rate $\eta_{\theta}$  
**Output**: trained parameter $\Theta$ for the neural estimator  
1: **repeat**  
2: &nbsp;&nbsp;&nbsp;&nbsp; **for** $n = 1,\dots ,N_{\mathrm{batch}}$ **do**  
3: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; generate a query point $\tilde{X}_n\sim \mathrm{Uniform}(\mathbb{X})$  
4: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; find the $k$ nearest neighbors of $\tilde{X}_n$ from data $(X_{m})_{m = 1}^{M}$ and collect accordingly $(\tilde{Y}_{n,i})_{i = 1}^{k}$  
5: &nbsp;&nbsp;&nbsp;&nbsp; **end for**  
6: &nbsp;&nbsp;&nbsp;&nbsp; compute with Sinkhorn algorithm (Y is equipped with $\| \cdot \| _1$):  
&nbsp;&nbsp;&nbsp;&nbsp; $L[\theta ]\coloneqq \sum_{n = 1}^{N_{\mathrm{batch}}}\mathcal{W}\left(\frac{1}{k}\sum_{i = 1}^{k}\delta_{\tilde{Y}_{n,i}},\frac{1}{N_{\mathrm{atom}}}\sum_{j = 1}^{N_{\mathrm{atom}}}\delta_{y_j^\theta (\tilde{X}_n)}\right)$ (11)  
7: &nbsp;&nbsp;&nbsp;&nbsp; update $\theta \gets \theta -\eta_{\theta}\nabla_{\theta}L[\theta ]$  
8: **until** Convergence  
9: **return** $\Theta = \theta$

### 3.1 Overview of key components

In this section, we outline the three key components of our implementation. Each of these components addresses a specific issue:

- Managing the computational cost arising from the nearest neighbors search.  
- Implementing gradient descent after computing $\mathcal{W}$  
- Selecting an appropriate Lipschitz constant for the neural estimator, preferably at a local level.

Further details and ablation analysis on these three components can be found in Section 5.

#### 3.1.1 APPROXIMATE NEAREST NEIGHBORS SEARCH WITH RANDOM BINARY SPACE PARTITIONING (ANNS-RBSP)

Given a query point, performing an exact search for its $k$-nearest-neighbor requires $O(M)$ operations. While a single search is not overly demanding, executing multiple searches as outlined in Algorithm 1 can result in significant computational time, even when leveraging GPU-accelerated parallel computing. To address this, we use ANNS-RBSP as a more cost-effective alternative. Prior to searching, we sort $(X_{m})_{m = 1}^{M}$ along each axis and record the order of indices. During the search, the data is divided into smaller subsets by repeatedly applying bisection on these sorted indices, with a random bisecting ratio, on a randomly chosen axis. Furthermore, we apply a restriction that mandates bisection along the longest edge of a rectangle when the edge ratio exceeds certain value (a hyper-parameter of the model). We record the bounding rectangle for each subset created through this partitioning process. Once partitioning is complete, we generate a small batch of query points within each rectangle and identify the $k$ nearest neighbors for each query point within that same rectangle. For a visual representation of ANNS-BSP, we refer to Figure 1. Leveraging the sorted indices, we can reapply this partitioning method during every training episode without much computational cost. We refer to Section 5.1 for additional details. There are similar ideas in the extant literature (cf. Hajebi et al. (2011); Ram and Sinha (2019); Li et al. (2020a)). Given the substantial differences in our setting, however, we conduct further empirical analysis in Section 5.1 to showcase the advantage of our approach against exact search.

![](images/4eb957e305a3677a621da0614fbe43fb9d3153210a24f0ea4640bef41065f2c0.jpg)  
**Figure 1**: An instance of RBSP in $[0,1]^2$.  
The 2D unit box is partitioned into 16 rectangles based on 500 samples from Uniform([0, 1]). Note that the overlap between the bounding rectangles is intentionally maintained. Each partitioning is performed along an axis selected at random, dividing the samples within the pre-partitioned rectangle according to a random ratio drawn from Uniform([0.45, 0.55]). The edge ratio for mandatory bisecting along the longest edge is 5. If this ratio is exceeded, partitioning along the longest edge is enforced. The black dots represent samples within the respective rectangle.

#### 3.1.2 COMPUTING $\mathcal{W}$ FOR GRADIENT DESCENT

The following discussion pertains to the computation of (11), with the subsequent gradient descent in consideration. For simplicity, let us focus on the summand and reduce the problem to the following minimization. Let $(\tilde{y}_1,\dots ,\tilde{y}_k)\in \mathbb{Y}^k$ be fixed, we aim to find

$$
\underset {y \in \mathbb {Y} ^ {n}} {\arg \min } \mathcal {W} \left(\frac {1}{k} \sum_ {i = 1} ^ {k} \delta_ {\tilde {y} _ {i}}, \frac {1}{n} \sum_ {j = 1} ^ {n} \delta_ {y _ {j}}\right). \tag {12}
$$

The criterion in (12) is convex as $\mathcal{W}$ is convex in both arguments (cf. (Villani, 2008, Theorem 4.8)). To solve (12), as is standard, we cast it into a discrete optimal transport problem. To do so, first introduce the $(k\times n)$-cost matrix $\mathsf{C}_y$, where $\mathsf{C}_{y,ij} \coloneqq \| \tilde{y}_i - y_j\|_1$. As the criterion in (12) has uniform weights on the atoms, we next aim to solve the problem

$$
\underset {\mathsf {T} \in [ 0, 1 ] ^ {k \times n}} {\arg \min } \left\{\varphi_ {y} (\mathsf {T}) := \sum_ {(i, j) \in \{1, \dots , k \} \times \{1, \dots , n \}} \mathsf {T} _ {i j} \mathsf {C} _ {y, i j} \right\} \tag {13}
$$

$$
\mathrm {subject to} \quad \sum_ {j = 1} ^ {n} \mathsf {T} _ {i j} = \frac {1}{k}, i = 1, \ldots , k \quad \mathrm {and} \quad \sum_ {i = 1} ^ {k} \mathsf {T} _ {i j} = \frac {1}{n}, j = 1, \ldots , n.
$$

Let $\mathsf{T}_y^*$ be an optimal transport plan that solves (13) for $y$ fixed. Taking derivative of $y\mapsto \varphi_y(\cdot)$ yields

$$
\left. \partial_ {y _ {j}} \varphi_ {y} (\mathsf {T}) \right| _ {\mathsf {T} = \mathsf {T} _ {y} ^ {*}} = \sum_ {i \in \{1, \dots , k \}} \mathsf {T} _ {y, i j} ^ {*} \partial_ {y _ {j}} \| \tilde {y} _ {i} - y _ {j} \| _ {1}, \quad j = 1, \dots , n. \tag {14}
$$

This gradient is in general not the gradient corresponding to (12), as $\mathsf{T}_y^*$ depends on $y$, while (14) excludes such dependence. Nevertheless, it is still viable to update $y$ using the gradient descent that employs the partial gradient specified in (14). To justify this update rule, first consider $y' \in \mathbb{Y}$ satisfying $\varphi_{y'}(\mathsf{T}_y^*) \leq \varphi_y(\mathsf{T}_y^*)$, then observe that

$$
\mathcal {W} \left(\frac {1}{k} \sum_ {i = 1} ^ {k} \delta_ {\tilde {y} _ {i}}, \frac {1}{n} \sum_ {j = 1} ^ {n} \delta_ {y _ {j} ^ {\prime}}\right) \leq \varphi_ {y ^ {\prime}} (\mathsf {T} _ {y} ^ {*}) \leq \varphi_ {y} (\mathsf {T} _ {y} ^ {*}) = \mathcal {W} \left(\frac {1}{k} \sum_ {i = 1} ^ {k} \delta_ {\tilde {y} _ {i}}, \frac {1}{n} \sum_ {j = 1} ^ {n} \delta_ {y _ {j}}\right).
$$

This inequality is strict if $\varphi_{y'}(\mathsf{T}_y^*) < \varphi_y(\mathsf{T}_y^*)$. We refer to (Peyre and Cuturi, 2019, Section 9.1) and the reference therein for related discussions.

The Sinkhorn algorithm, which adds an entropy regularization, is a widely-used algorithm for approximating the solution to (13). Specifically, here, it is an iterative scheme that approximately solves the following regularized problem, subject to the constraints in (13),

$$
\underset {T ^ {\epsilon} \in [ 0, 1 ] ^ {k \times n}} {\arg \min } \left\{\sum_ {i, j \in \{1, \dots , k \} \times \{i, \dots , n \}} T _ {i j} ^ {\epsilon} C _ {i j} + \epsilon \sum_ {i, j \in \{1, \dots , k \} \times \{i, \dots , n \}} T _ {i j} ^ {\epsilon} (\log T _ {i j} - 1) \right\}, \tag {15}
$$

where $\epsilon > 0$ is a hyper-parameter, and should not be confused with the $\varepsilon$ used elsewhere. We refer to Section 5.2 for further details. We also refer to (Peyre and Cuturi, 2019, Section 4) and the reference therein for convergence analysis of the Sinkhorn algorithm. It is well known that the regularization term in (15) is related to the entropy of a discrete random variable. Larger values of $\epsilon$ encourages the regularized optimal transport plan to be more diffusive. That is, for larger values of $\epsilon$, the mass from each $y_{j}$ is distributed more evenly across all $\tilde{y}_i$'s. Performing gradient descent along the direction in (14) tends to pull $y_{j}$'s towards the median of the $\tilde{y}_i$'s, as we are equipping $\mathbb{Y}$ with the norm $\| \cdot \|_1$. Conversely, small values of $\epsilon$ often leads to instability, resulting in NaN loss/gradient. To help with these issues, we implement the Sinkhorn algorithm after normalizing the cost matrix. Additionally, we use a large $\epsilon$ (e.g., 1) in the first few training episodes, then switch to a smaller $\epsilon$ (e.g., 0.1) in later episodes. Furthermore, we impose sparsity on the transport plan by manually setting the smaller entries of the transport plan to 0. The specific detailed configurations and related ablation analysis are provided in Section 5.2 and Appendix C.

#### 3.1.3 NETWORK STRUCTURE THAT INDUCES LOCALLY ADAPTIVE LIPSCHITZ CONTINUITY

As previously discussed, it is desirable for the neural estimator to exhibit certain Lipschitz continuity. In practice, however, determining an appropriate Lipschitz constant for training the neural estimator $\tilde{P}^{\theta}$ is challenging, largely because understanding the true Lipschitz continuity of $P$ (if it exists) in a data-driven manner is very challenging. Additionally, the estimate provided in Proposition 13 is probabilistic. Fortunately, a specific network structure allows the neural estimator, when properly trained, to exhibit locally adaptive Lipschitz continuity. Subsequently, we provide a high-level overview of this network structure. Further detailed configurations and ablation analysis are presented in Section 5.3 and Appendix C.

Consider a fully connected feed-forward neural network with equal width hidden layers and layer-wise residual connection (He et al. (2016)). Let $N_{\mathrm{neuron}}$ denote the width of the hidden layers. For activation, we use Exponential Linear Unit (ELU) function (Clevert et al. (2016)), denoted by $\sigma$. For hidden layers, we employ the convex potential layer introduced in Meunier et al. (2022),

$$
\mathsf {x} _ {\mathrm {o u t}} = \mathsf {x} _ {\mathrm {i n}} - \| \mathsf {W} \|