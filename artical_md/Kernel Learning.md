# Learning conditional distributions on continuous spaces


# Abstract

We investigate sample-based learning of conditional distributions on multi-dimensional unit boxes, allowing for different dimensions of the feature and target spaces. Our approach involves clustering data near varying query points in the feature space to create empirical measures in the target space. We employ two distinct clustering schemes: one based on a fixed-radius ball and the other on nearest neighbors. We establish upper bounds for the convergence rates of both methods and, from these bounds, deduce optimal configurations for the radius and the number of neighbors. We propose to incorporate the nearest neighbors method into neural network training, as our empirical analysis indicates it has better performance in practice. For efficiency, our training process utilizes approximate nearest neighbors search with random binary space partitioning. Additionally, we employ the Sinkhorn algorithm and a sparsity-enforced transport plan. Our empirical findings demonstrate that, with a suitably designed structure, the neural network has the ability to adapt to a suitable level of Lipschitz continuity locally. For reproducibility, our code is available at https://github.com/zcheng-a/LCD_kNN.

Keywords: non-parametric statistics, Wasserstein distance, deep learning, Lipschitz continuity

# 1. Introduction

Learning the conditional distribution is a crucial aspect of many decision-making scenarios. While this learning task is generally challenging, it presents unique complexities when explored in a continuous space setting. Below, we present a classic example (cf. Booth et al. (1992); Pflug and Pichler (2016)) that highlights this core challenge.

For simplicity, we suppose the following model

$$
Y = \frac {1}{2} X + \frac {1}{2} U,
$$

where the feature variable  $X$  and the noise  $U$  are independent Uniform([0,1]), and  $Y$  is the target variable. Upon collecting a finite number of independent samples  $\mathcal{D} = \{(X_m,Y_m)\}_{m=1}^M$ , we aim to estimate the conditional distribution of  $Y$  given  $X$ . Throughout, we treat this conditional distribution as a measure-valued function of  $x$ , denoted by  $P_x$ . A naive approach is to first form an empirical joint measure

$$
\hat {\psi} := \frac {1}{M} \sum_ {m = 1} ^ {M} \delta_ {(X _ {m}, Y _ {m})},
$$

where  $\delta$  stands for the Dirac measure, and then use the conditional distribution induced from  $\hat{\psi}$  as an estimator. As the marginal distribution of  $X$  is continuous, with probability 1 (as  $\mathbb{P}(X_m = X_{m'}) = 0$  for all  $m \neq m'$ ), we have that

$$
\widehat {P} _ {x} = \left\{ \begin{array}{l l} \delta_ {Y _ {m}}, & x = X _ {m} \text {f o r s o m e} m, \\ \text {U n i f o r m} ([ 0, 1 ]), & \text {o t h e r w i s e}. \end{array} \right.
$$

Regardless of the sample size  $M$ ,  $\widehat{P}_x$  fails to approximate the true conditional distribution,

$$
P _ {x} = \mathrm {U n i f o r m} \left([ x, x + \frac {1}{2} ]\right), \quad x \in [ 0, 1 ].
$$

Despite the well-known convergence of the (joint) empirical measure to the true distribution Dudley (1969); Fournier and Guillin (2015), the resulting conditional distribution often fails to provide an accurate approximation of the true distribution. This discrepancy could be due to the fact that calculating conditional distribution is an inherently unbounded operation. As a remedy, clustering is a widely employed technique. Specifically, given a query point  $x$  in the feature space, we identify samples where  $X_{m}$  is close to  $x$  and use the corresponding  $Y_{m}$ 's to estimate  $P_{x}$ . Two prominent methods within the clustering approach are the kernel method and the nearest neighbors method<sup>2</sup>. Roughly speaking, the kernel method relies primarily on proximity to the query point for selecting  $X_{m}$ 's, while the nearest neighbors method focuses on the rank of proximity. Notably, discretizing the feature space (also known as quantization), a straightforward yet often effective strategy, can be seen as a variant of the kernel method with static query points and flat kernels.

The problem of estimating conditional distributions can be addressed within the non-parametric regression framework, by employing clustering or resorting to non-parametric least squares, among others. Alternatively, it is feasible to estimate the conditional density function directly: a widely-used method involves estimating the joint and marginal density functions using kernel smoothing and then calculating their ratio. This method shares similarities with the clustering heuristics mentioned earlier. For a more detailed review of these approaches, we refer to Section 1.2.

This work draws inspiration from recent advancements in estimating discrete-time stochastic processes using conditional density function estimation (Pflug and Pichler (2016)) and quantization methods (Backhoff et al. (2022); Acciaio and Hou (2023)). A notable feature of these works is their use of the Wasserstein distance to calculate local errors: the difference between the true and estimated conditional distributions at a query point  $x$ . One could average these local errors across

different values of  $x$ 's to gauge the global error. Employing Wasserstein distances naturally frames the study within the context of weak convergence, thereby enabling discussions in a relatively general setting, although this approach may yield somewhat weaker results in terms of the mode of convergence. Moreover, utilizing a specific distance rather than the general notion of weak convergence enables a more tangible analysis of the convergence rates and fluctuations. We would like to point out that the advancements made in Backhoff et al. (2022); Acciaio and Hou (2023), as well as our analysis in this paper, relies on recent developments concerning the Wasserstein convergence rate of empirical measures under i.i.d. sampling from a static distribution (cf. Fournier and Guillin (2015)).

# 1.1 Main contributions

First, we introduce some notations to better illustrate the estimators that we study. Let  $\mathbb{X}$  and  $\mathbb{Y}$  be multi-dimensional unit cubes, with potentially different dimensions, for feature and target spaces. For any integer  $M\geq 1$ , any  $\mathsf{D} = \{(x_m,y_m)\}_{m = 1}^M\in (\mathbb{X}\times \mathbb{Y})^M$ , and any Borel set  $A\subset \mathbb{X}$ , we define a probability measure on  $\mathbb{Y}$  by

$$
\hat {\mu} _ {A} ^ {\mathrm {D}} := \left\{ \begin{array}{l l} \left(\sum_ {m = 1} ^ {M} \mathbb {1} _ {A} \left(x _ {m}\right)\right) ^ {- 1} \sum_ {m = 1} ^ {M} \mathbb {1} _ {A} \left(x _ {m}\right) \delta_ {y _ {m}}, & \sum_ {m = 1} ^ {M} \mathbb {1} _ {A} \left(x _ {m}\right) > 0, \\ \lambda_ {\mathbb {Y}}, & \text {o t h e r w i s e}, \end{array} \right. \tag {1}
$$

where  $\lambda_{\mathbb{Y}}$  is the Lebesgue measure on  $\mathbb{Y}$  and, for  $y \in \mathbb{Y}$ ,  $\delta_y$  is a Dirac measure with atom at  $y$ . In general, one could consider weighting  $\delta_{y_m}$ 's (cf. (Györfi et al., 2002, Section 5), (Biau and Devroye, 2015, Chapter 5)), which may offer additional benefits in specific applications. As such adjustments are unlikely to affect the convergence rate, however, we use uniform weighting for simplicity.

With (random) data  $\mathcal{D} = \{(X_m, Y_m)\}_{m=1}^M$ , we aim to estimate the conditional distribution of  $Y$  given  $X$ . We view this conditional distribution as a measure-valued function  $P: \mathbb{X} \to \mathcal{P}(\mathbb{Y})$  and use a subscript for the input argument and write  $P_x$ . Consider a clustering scheme<sup>3</sup> given by the map  $\mathcal{A}^{\mathsf{D}}: \mathbb{X} \to 2^{\mathbb{X}}$ . We investigate estimators of the form  $x \mapsto \hat{\mu}_{\mathcal{A}^{\mathcal{D}}(x)}^{\mathcal{D}}$ . We use  $\widehat{P^A}$  to denote said estimator and suppress  $\mathcal{D}$  from the notation for convenience. In later sections, we consider two kinds of maps  $\mathcal{A}^{\mathsf{D}}$  (i) a ball with fixed radius centered at  $x$ , called an  $r$ -box and (ii) the  $k$ -nearest neighbors of  $x$ , called  $k$ -nearest-neighbor estimator. See Definitions 5 and 9 for more details.

One of our main contribution pertains to analyzing the error

$$
\int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \widehat {P} _ {x} ^ {\mathcal {A}}\right) \nu (\mathrm {d} x), \tag {2}
$$

where  $\mathcal{W}$  is the 1-Wasserstein distance (see (Villani, 2008, Particular Case 6.2)) and  $\nu \in \mathcal{P}(\mathbb{X})$  is arbitrary and provides versatility to the evaluation criterion. A canonical choice for  $\nu$  is the Lebesgue measure on  $\mathbb{X}$ , denoted by  $\lambda_{\mathbb{X}}$ . This is particularly relevant in control settings where  $\mathbb{X}$  represents the state-action space and accurate approximations across various state and action scenarios are crucial for making informed decisions. The form of error above is also foundational in stochastic process estimation under the adapted Wasserstein distance (cf. (Backhoff et al., 2022, Lemma 3.1)), making the techniques we develop potentially relevant in other contexts. Under the assumption that  $P$  is Lipschitz continuous (Assumption 2) and standard assumptions on the data collection process (Assumption 3), we analyze the convergence rate and fluctuation by bounding

the following two quantities

$$
\mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \widehat {P} _ {x} ^ {\mathcal {A}}\right) \nu (\mathrm {d} x) \right] \quad \mathrm {a n d} \quad \mathrm {V a r} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \widehat {P} _ {x} ^ {\mathcal {A}}\right) \nu (\mathrm {d} x) \right].
$$

Moreover, by analyzing the above quantities, we gain insights into the optimal choice of the clustering mapping  $\mathcal{A}$ . For the detail statements of these results, we refer to Theorems 7, 10, 8, and 11. We also refer to Section 2.4 for related comments.

To illustrate another aspect of our contribution, we note by design  $x \mapsto \widehat{P}_x^{\mathcal{A}}$  is piece-wise constant. This characteristic introduces limitations. Notably, it renders the analysis of performance at the worst-case  $x$  elusive. Contrastingly, by building a Lipschitz-continuous parametric estimator  $\tilde{P}^\Theta$  from the raw estimator  $\widehat{P}^{\mathcal{A}}$ , in Proposition 13 we demonstrate that an upper bound on the aforementioned expectation allows us to derive a worst-case performance guarantee. Guided by Proposition 13, we explore a novel approach of training a neural network for estimation, by using  $\widehat{P}^{\mathcal{A}}$  as training data and incorporating suitably imposed Lipschitz continuity. To be comprehensive, we include in Section 1.2 a review of studies on Lipschitz continuity in neural networks.

In Section 3.1, we define  $\tilde{P}^{\theta}$  as a neural network that approximates  $P$ , where  $\theta$  represents the network parameters. We train  $\tilde{P}^{\theta}$  with the objective:

$$
\underset {\theta} {\arg \min} \sum_ {n = 1} ^ {N} \mathcal {W} \left(\widehat {P} _ {\tilde {X} _ {n}} ^ {\mathcal {A}}, \tilde {P} _ {\tilde {X} _ {n}} ^ {\theta}\right),
$$

where  $(\tilde{X}_n)_{n=1}^N$  is a set of randomly selected query points. For implementation purposes, we use the  $k$ -nearest-neighbor estimator in the place of  $\hat{P}^{\mathcal{A}}$  (see Definition 9). To mitigate the computational costs stemming from the nearest neighbors search, we employ the technique of Approximate Nearest Neighbor Search with Random Binary Space Partitioning (ANN-RBSP), as discussed in Section 3.1.1. In Section 3.1.2, we compute  $\mathcal{W}$  using the Sinkhorn algorithm, incorporating normalization and enforcing sparsity for improved accuracy. To impose a suitable level of local Lipschitz continuity on  $\tilde{P}^\theta$ , in Section 3.1.3, we employ a neural network with a specific architecture and train the networks using a tailored procedure. The key component of this architecture is the convex potential layer introduced in Meunier et al. (2022). In contrast to most extant literature that imposes Lipschitz continuity on neural networks, our approach does not utilize specific constraint or regularization of the objective function, but relies on certain self-adjusting mechanism embedded in the training.

In Section 3.2, we evaluate the performance of the trained  $\tilde{P}^{\theta}$ , denoted by  $\tilde{P}^{\Theta}$ , using three sets of synthetic data in 1D and 3D spaces. Our findings indicate that  $\tilde{P}^{\Theta}$  generally outperforms  $\widehat{P}^{\mathcal{A}}$ , even though it is initially trained to match  $\widehat{P}^{\mathcal{A}}$ . This superior performance persists even when comparing  $\tilde{P}^{\Theta}$  to different  $\widehat{P}^{\mathcal{A}}$  using various clustering parameters, without retraining  $\tilde{P}^{\Theta}$ . Furthermore, despite using the same training parameters,  $\tilde{P}^{\Theta}$  consistently demonstrates the ability to adapt to a satisfactory level of local Lipschitz continuity across all cases. Moreover, in one of the test cases, we consider a kernel that exhibits a jump discontinuity, and we find that  $\tilde{P}^{\Theta}$  handles this jump case well despite Lipschitz continuity does not hold.

Lastly, we provide further motivation of our approach by highlighting some potential applications for  $\tilde{P}^{\Theta}$ . The first application is in model-based policy gradient method in reinforcement learning. We anticipate that the enforced Lipschitz continuity allows us to directly apply the policy gradient update via compositions of  $\tilde{P}^{\Theta}$  and cost function for more effective optimality searching. The second application of  $\tilde{P}^{\Theta}$  is in addressing optimisation in risk-averse Markov decision processes, where dynamic programming requires knowledge beyond the conditional expectation of the risk-to-go (cf. Chow et al. (2015); Huang and Haskell (2017); Coache et al. (2023); Cheng and Jaimungal (2023)). The study of these applications is left for further research.

# 1.2 Related works

In this section, we will first review the clustering approach in estimating conditional distributions, and then proceed to review recent studies on Lipschitz continuity in neural networks.

# 1.2.1 ESTIMATING CONDITIONAL DISTRIBUTIONS VIA CLUSTERING

The problem of estimating conditional distributions is frequently framed as non-parametric regression problems for real-valued functions. For instance, when  $d_{\mathbb{Y}} = 1$ , estimate the conditional  $\alpha$ -quantile of  $Y$  given  $X$ . Therefore, we begin by reviewing some of the works in non-parametric regression.

The kernel method in non-parametric regression traces its origins back to the Nadaraya-Watson estimator (Nadaraya (1964); Watson (1964)), if not earlier. Subsequent improvements have been introduced, such as integral smoothing (Gasser and Müller (1979), also known as the Gasser-Müller estimator), local fitting with polynomials instead of constants (Fan (1992)), and adaptive kernels (Hall et al. (1999)). Another significant area of discussion is the choice of kernel bandwidth, as detailed in works like Hardle and Marron (1985); Gijbels and Goderniaux (2004); Kohler et al. (2014). Regarding convergence rates, analyses under various settings can be found in Stone (1982); Hall and Hart (1990); Kohler et al. (2009, 2010), with Stone (1982) being particularly relevant to our study for comparative purposes. According to Stone (1982), if the target function is Lipschitz continuous, with i.i.d. sampling and that the sampling distribution in the feature space has a uniformly positive density, then the optimal rate of the  $\| \cdot \| _1$ -distance between the regression function and the estimator is of the order  $M^{-\frac{1}{d_{\mathbb{X}} + 2}}$ . For a more comprehensive review of non-parametric regression using kernel methods, we refer to the books such as Györfi et al. (2002); Ferraty and Vieu (2006); Wasserman (2006) and references therein.

Non-parametric regression using nearest neighbors methods originated from classification problems (Fix and Hodges (1951)). Early developments in this field can be found in Mack (1981); Devroye (1982); Bhattacharya and Gangopadhyay (1990). For a comprehensive introduction to nearest neighbors methods, we refer to Györfi et al. (2002). More recent reference such as Biau and Devroye (2015) offers further detailed exploration of the topic. The nearest neighbor method can be viewed as a variant of the kernel method that adjusts the bandwidth based on the number of local data points—a property that has gained significant traction. Recently, the application of the nearest neighbor method has expanded into various less standard settings, including handling missing data (Rachdi et al. (2021)), reinforcement learning (Shah and Xie (2010); Giegrich et al. (2024)), and time series forecasting (Martínez et al. (2017)). For recent advancements in convergence analysis beyond the classical setting, see Zhao and Lai (2019); Padilla et al. (2020); Ryu and Kim (2022); Demirkayaa et al. (2024).

Although the review above mostly focuses on clustering approach, other effective approaches exist, such as non-parametric least square, or more broadly, conditional elicitability (e.g., Györfi et al. (2002); Muandet et al. (2017); Wainwright (2019); Coache et al. (2023)). Non-parametric least square directly fits the data using a restricted class of functions. At first glance, this approach appears distinct from clustering. However, they share some similarities in their heuristics: the rigidity of the fitting function, due to imposed restrictions, allows data points near the query point to affect the estimation, thereby implicitly incorporating elements of clustering.

Apart from non-parametric regression, conditional density function estimation is another significant method for estimating conditional distributions. One approach is based on estimating joint and marginal density functions, and then using the ratio of these two to produce an estimator for the conditional density function. A key technique used in this approach is kernel smoothing. Employing a static kernel for smoothing results in a conditional density estimator that shares similar clustering

heuristics to those found in the kernel method of non-parametric regression. For a comprehensive overview of conditional density estimation, we refer to reference books such as Scott (2015); Simonoff (1996). For completeness, we also refer to (Muandet et al., 2017, Section 5.1) for a perspective on static density function estimation from the standpoint of reproducing kernel Hilbert space. Further discussions on estimation using adaptive kernels can be found in, for example, Bashtannyk and Hyndman (2001); Lacour (2007); Bertin et al. (2016); Zhao and Tabak (2023).

Despite extensive research in non-parametric regression and conditional density function estimation, investigations from the perspective of weak convergence have been relatively limited, only gaining more traction in the past decade. Below, we highlight a few recent studies conducted in the context of estimating discrete-time stochastic processes under adapted Wasserstein distance, as the essence of these studies are relevant to our evaluation criterion (2). Pflug and Pichler (2016) explores the problem asymptotically, employing tools from conditional density function estimation with kernel smoothing. Subsequently, Backhoff et al. (2022) investigates a similar problem with a hypercube as state space, employing the quantization method. Their approach removes the need to work with density functions. They calculate the convergence rate, by leveraging recent developments in the Wasserstein convergence rate of empirical measures Fournier and Guillin (2015). Moreover, a sub-Gaussian concentration with parameter  $M^{-1}$  is established. The aforementioned results are later extended to  $\mathbb{R}^d$  in Acciaio and Hou (2023), where a non-uniform grid is used to mitigate assumptions on moment conditions. Most recently, Hou (2024) examines smoothed variations of the estimators proposed in Backhoff et al. (2022); Acciaio and Hou (2023). Other developments on estimators constructed from smoothed quantization can be found in Smid and Kozmik (2024).

Lastly, regarding the machine learning techniques used in estimating conditional distributions, conditional generative models are particularly relevant. For reference, see Mirza and Osindero (2014); Papamakarios et al. (2017); Vaswani et al. (2017); Fetaya et al. (2020). These models have achieved numerous successes in image generation and natural language processing. We suspect that, due to the relatively discrete (albeit massive) feature spaces in these applications, clustering is implicitly integrated into the training procedure. In continuous spaces, under suitable setting, clustering may also become an embedded part of the training procedure. For example, implementations in Li et al. (2020b); Vuletic et al. (2024); Hosseini et al. (2024) do not explicitly involve clustering and use training objectives that do not specifically address the issues highlighted in the motivating example at the beginning of the introduction. Their effectiveness could possibly be attributed to certain regularization embedded within the neural network and training procedures. Nevertheless, research done in continuous spaces that explicitly uses clustering approaches when training conditional generative models holds merit. Such works are relatively scarce. For an example of this limited body of research, we refer to Xu and Acciaio (2022), where the conditional density function estimator from Pflug and Pichler (2016) is used to train an adversarial generative network for stochastic process generation.

# 1.2.2 LIPSCHITZ CONTINUITY IN NEURAL NETWORKS

Recently, there has been increasing interest in understanding and enforcing Lipschitz continuity in neural networks. The primary motivation is to provide a certifiable guarantee for classification tasks performed by neural networks: it is crucial that minor perturbations in the input object have a limited impact on the classification outcome.

One strategy involves bounding the Lipschitz constant of a neural network, which can then be incorporated into the training process. For refined upper bounds on the (global) Lipschitz constant, see, for example, Bartlett et al. (2017); Virmaux and Scaman (2018); Tsuzuki et al. (2018); Fazlyab et al. (2019); Xue et al. (2022); Fazlyab et al. (2024). For local bounds, we refer to Jordan and

Dimakis (2021); Bhowmick et al. (2021); Shi et al. (2022) and the references therein. We also refer to Zhang et al. (2022) for a study of the Lipschitz property from the viewpoint of boolean functions.

Alternatively, designing neural network architectures that inherently ensure desirable Lipschitz constants is another viable strategy. Works in this direction include Meunier et al. (2022); Singla et al. (2022); Wang and Manchester (2023); Araujo et al. (2023). Notably, the layer introduced in Meunier et al. (2022) belongs to the category of residual connection (He et al. (2016)).

Below, we review several approaches that enforce Lipschitz constants during neural network training. Tsuzuku et al. (2018); Liu et al. (2022) explore training with a regularized objective function that includes upper bounds on the network's Lipschitz constant. Gouk et al. (2021) frame the training problem into constrained optimization and train with projected gradients descent. Given the specific structure of the refined bound established in Fazlyab et al. (2019), Pauli et al. (2022) combines training with semi-definite programming. They develop a version with a regularized objective function and another that enforces the Lipschitz constant exactly. Fazlyab et al. (2024) also investigates training with a regularized objective but considers Lipschitz constants along certain directions. Huang et al. (2021) devises a training procedure that removes components from the weight matrices to achieve smaller local Lipschitz constants. Trockman and Kolter (2021) initially imposes orthogonality on the weight matrices, and subsequently enforces a desirable Lipschitz constant based on that orthogonality. Ensuring desirable Lipschitz constants with tailored architectures, Singla et al. (2022); Wang and Manchester (2023) train the networks directly. Although the architecture proposed in Meunier et al. (2022) theoretically ensures the Lipschitz constant, it requires knowledge of the spectral norm of the weight matrices, which does not admit explicit expression in general. Their training approach combines power iteration for spectral norm approximation with the regularization methods used in Tsuzuku et al. (2018).

Finally, we note that due to their specific application scenarios, these implementations concern relatively stringent robustness requirements and thus necessitate more specific regularization or constraints. In our setting, it is generally desirable for the neural network to automatically adapt to a suitable level of Lipschitz continuity based on the data, while also avoiding excessive oscillations from over-fitting. The literature directly addressing this perspective is limited (especially in the setting of conditional distribution estimation). We refer to Bai et al. (2021); Bountakas et al. (2023); Cohen et al. (2019) for discussions that could be relevant.

# 1.3 Organization of the paper

Our main theoretical results are presented in Section 2. Section 3 is dedicated to the training of  $\tilde{P}^{\Theta}$ . We will outline the key components of our training algorithm and demonstrate its performance on three sets of synthetic data. We will prove the theoretical results in Section 4. Further implementation details and ablation analysis are provided in Section 5. In Section 6, we discuss the weaknesses and potential improvements of our implementation. Appendix B and C respectively contain additional plots and a table that summarizes the configuration of our implementation. Additionally, Appendix D includes a rougher version of the fluctuation results.

# Notations and terminologies

Throughout, we adopt the following set of notations and terminologies.

On any normed space  $(E, \| \cdot \|$ $B(x,\gamma)$  denotes the closed ball of radius  $\gamma$  around  $x$ , namely  $B(x,\gamma) = \{x' \in E \mid \| x - x' \| \leq \gamma\}$ .  
For any measurable space  $(E,\mathcal{E})$ ,  $\mathcal{P}(E)$  denotes the set of probability distributions on  $(E,\mathcal{E})$ . For all  $x\in E$ ,  $\delta_x\in \mathcal{P}(E)$  denotes the Dirac mass at  $x$ .

$\circ$  We endow normed spaces  $(E, \| \cdot \|$ $^\circ$  with their Borel sigma-algebra  $\mathcal{B}(E)$ , and  $\mathcal{W}$  denotes the 1-Wasserstein distance on  $\mathcal{P}(E)$ .  
On  $\mathbb{X} = [0,1]^d$ , we denote by  $\lambda_{\mathbb{X}}$  the Lebesgue measure. We say a measure  $\nu \in \mathcal{P}(\mathbb{X})$  is dominated by Lebesgue measure with a constant  $\overline{C} > 0$  if  $\nu(A) \leq \overline{C}\lambda_{\mathbb{X}}(A)$  for all  $A \in \mathcal{B}([0,1]^d)$ .  
$\circ$  The symbol  $\sim$  denotes equivalence in the sense of big O notation, indicating that each side dominates the other up to a multiplication of some positive absolute constant. More precisely,  $a_{n} \sim b_{n}$  means there are finite constants  $c, C > 0$  such that

$$
c a _ {n} \leq b _ {n} \leq C a _ {n}, \quad n \in \mathbb {N}.
$$

Similarly,  $\lesssim$  implies that one side is of a lesser or equal, in the sense of big O notation, compared to the other.

# 2. Theoretical results

In Section 2.1, we first formally set up the problem and introduce some technical assumption. We then study in Section 2.2 and 2.3 the convergence and fluctuation of two versions of  $\hat{P}^{\mathcal{A}}$ , namely, the  $r$ -box estimator and the  $k$ -nearest-neighbor estimator. Related comments are organized in Section 2.4. Moreover, in Section 2.5, we provide a theoretical motivation for the use of  $\tilde{P}^{\Theta}$ , the Lipschitz-continuous parametric estimator trained from  $\hat{P}^{\mathcal{A}}$ .

# 2.1 Setup

For  $d_{\mathbb{X}}, d_{\mathbb{Y}} \geq 1$  two integers, we consider  $\mathbb{X} := [0,1]^{d_{\mathbb{X}}}$  and  $\mathbb{Y} := [0,1]^{d_{\mathbb{Y}}}$ , endowed with their respective sup-norm  $\| \cdot \|_{\infty}$ .

Remark 1 The sup-norm is chosen for simplicity of the theoretical analysis only: as all norms on  $\mathbb{R}^n$  are equivalent (for any generic  $n\geq 1$ ), our results are valid, up to different multiplicative constants, for any other choice of norm.

We aim to estimate an unknown probabilistic kernel

$$
\begin{array}{l} P: \mathbb {X} \to \mathcal {P} (\mathbb {Y}) \\ x \mapsto P _ {x} (\mathrm {d} y). \\ \end{array}
$$

To this end, given an integer-valued sample size  $M \geq 1$ , we consider a set of (random) data points  $\mathcal{D} := \{(X_m, Y_m)\}_{m=1}^M$  associated to  $P$ . We also define the set of projections of the data points onto the feature space as  $\mathcal{D}_{\mathbb{X}} := \{X_m\}_{m=1}^M$ .

Throughout this section, we work under the following technical assumptions.

Assumption 2 (Lipschitz continuity of kernel) There exists  $L \geq 0$  such that, for all  $(x, x') \in \mathbb{X}^2$ ,

$$
\mathcal {W} \left(P _ {x}, P _ {x ^ {\prime}}\right) \leq L \| x - x ^ {\prime} \| _ {\infty}.
$$

Assumption 3 The following is true:

(i)  $\mathcal{D}$  is i.i.d. with probability distribution  $\psi \coloneqq \xi \otimes P$ , where  $\xi \in \mathcal{P}(\mathbb{X})$  and where  $\xi \otimes P \in \mathcal{P}(\mathbb{X} \times \mathbb{Y})$  is (uniquely, by Caratheodory extension theorem) defined by

$$
(\xi \otimes P) (A \times B) := \int_ {\mathbb {X}} \mathbb {1} _ {A} (x) P _ {x} (B) \xi (\mathrm {d} x), \quad A \in \mathcal {B} (\mathbb {X}), B \in \mathcal {B} (\mathbb {Y}).
$$

(ii) There exists  $\underline{c} \in (0,1]$  such that, for all  $A \in \mathcal{B}(\mathbb{X})$ ,  $\xi(A) \geq \underline{c} \lambda_{\mathbb{X}}(A)$ .

These assumptions allow us to analyze convergence and gain insights into the optimal clustering hyper-parameters without delving into excessive technical details. Assumption 2 is mainly used for determining the convergence rate. If the convergence rate is not of concern, it is possible to establish asymptotic results with less assumptions. We refer to Devroye (1982); Backhoff et al. (2022) for relevant results. The conditions placed on  $\xi$  in Assumption 3 are fairly standard, though less stringent alternatives are available. For instance, Assumption 3 (i) can be weakened by considering suitable dependence Hall and Hart (1990) or ergodicity in the context of stochastic processes Rudolf and Schweizer (2018). Assumption 3 (ii), implies there is mass almost everywhere and is aligned with the motivation from control settings discussed in the introduction. Assumptions 2 and 3 are not exceedingly stringent and provides a number of insights into the estimation problem. More general settings are left for further research.

The estimators discussed in subsequent sections are of the form  $\widehat{P^{\mathcal{A}}}$ , as introduced right after (1), for two specific choices of clustering schemes  $\mathcal{A}$  constructed with the data  $\mathcal{D}$ .

Remark 4 In the following study, we assert all the measurability needed for  $\widehat{P^A}$  to be well-defined. These measurability can be verified using standard measure-theoretic tools listed in, for example, (Aliprantis and Border, 2006, Section 4 and 15).

# 2.2 Results on  $r$ -box estimator

The first estimator, which we term the  $r$ -box estimator, is defined as follows.

Definition 5 Choose  $r$ , a real number, s.t.  $0 < r < \frac{1}{2}$ . The  $r$ -box estimator for  $P$  is defined by

$$
\begin{array}{l} \hat {P} ^ {r}: \mathbb {X} \rightarrow \mathcal {P} (\mathbb {Y}) \\ x \mapsto \hat {P} _ {x} ^ {r} := \hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D}}, \\ \end{array}
$$

where, for all  $x \in \mathbb{X}$ ,  $\mathcal{B}^r(x) \coloneqq B(\beta^r(x), r)$  and  $\beta^r(x) \coloneqq r \vee x \wedge (1 - r)$ , where  $r \vee \cdot$  and  $\cdot \wedge (1 - r)$  are applied entry-wise.

Remark 6 The set  $\mathcal{B}^r (x)$  is defined such that it is a ball of radius around  $x$  whenever  $x$  is at least  $r$  away from the boundary  $\partial \mathbb{X}$  (in all of its components), otherwise, we move the point  $x$  in whichever components are within  $r$  from  $\partial \mathbb{X}$  to be a distance  $r$  away from  $\partial \mathbb{X}$ . Consequently, for all  $0 < r < \frac{1}{2}$  and for all  $x\in \mathbb{X}$ ,  $\mathcal{B}^r (x)$  has a bona fide radius of  $r$ , as the center  $\beta^r (x)$  is smaller or equal to  $r$  away from  $\partial \mathbb{X}$ .

For the  $r$ -box estimator, we have the following convergence results. The theorem below discusses the convergence rate of the average Wasserstein distance between the unknown kernel evaluated at any point and its estimator, when the radius  $r$  is chosen optimally with respect to the data sample  $M$ . Section 4.2 is dedicated to its proof.

Theorem 7 Under Assumptions 2 and 3, choose  $r$  as follows

$$
r \sim \left\{ \begin{array}{l l} M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}}, & d _ {\mathbb {Y}} = 1, 2 \\ M ^ {- \frac {1}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}}, & d _ {\mathbb {Y}} \geq 3. \end{array} \right.
$$

Then, there is a constant  $C > 0$  (which depends only on  $d_{\mathbb{X}}, d_{\mathbb{Y}}, L, \underline{c}$ ), such that, for all probability distribution  $\nu \in \mathcal{P}(\mathbb{X})$ , we have

$$
\mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \hat {P} _ {x} ^ {r}\right) \nu (\mathrm {d} x) \right] \leq \sup  _ {x \in \mathbb {X}} \mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \hat {P} ^ {r}\right) \right] \leq C \times \left\{ \begin{array}{l l} M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}}, & d _ {\mathbb {Y}} = 1, \\ M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}} \ln (M), & d _ {\mathbb {Y}} = 2, \\ M ^ {- \frac {1}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}}, & d _ {\mathbb {Y}} \geq 3. \end{array} \right. \tag {3}
$$

Next, we bound the associated variance whose proof is postponed to Section 4.3.

Theorem 8 Under Assumptions 3, consider  $r \in (0, \frac{1}{2}]$ . Let  $\nu \in \mathcal{P}(\mathbb{X})$  be dominated by  $\lambda_{\mathbb{X}}$  with a constant  $\overline{C} > 0$ . Then,

$$
\mathrm {V a r} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \hat {P} _ {x} ^ {r}\right) \mathrm {d} \nu (x) \right] \leq \frac {4 ^ {d _ {\mathbb {X}} + 1} \overline {{C}} ^ {2}}{\underline {{c}} ^ {2} (M + 1)}.
$$

# 2.3 Results on  $k$ -nearest-neighbor estimator

Here, we focus in the second estimator - the  $k$ -nearest-neighbor estimator, defined as follows.

Definition 9 Let  $k \geq 1$  an integer. The  $k$ -nearest-neighbor estimator for  $P$  is defined by

$$
\check {P} ^ {k}: \mathbb {X} \rightarrow \mathcal {P} (\mathbb {Y})
$$

$$
x \mapsto \check {P} _ {x} ^ {k} := \hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}} (x)}} ^ {\mathcal {D}},
$$

where, for any integer  $M \geq 1$  and any  $\mathsf{D}_{\mathbb{X}} \in \mathbb{X}^M$ ,  $\mathcal{N}^{k,\mathsf{D}_{\mathbb{X}}}(\boldsymbol{x})$  contains (exactly)  $k$  points of  $\mathsf{D}_{\mathbb{X}}$  which are closest to  $x$ , namely

$$
\mathcal {N} ^ {k, \mathrm {D} _ {\mathbb {X}}} (x) := \left\{x ^ {\prime} \in \mathrm {D} _ {\mathbb {X}} \mid \| x - x ^ {\prime} \| _ {\infty} i s a m o n g t h e k - s m a l l e s t o f \left(\| x - x ^ {\prime} \| _ {\infty}\right) _ {x ^ {\prime} \in \mathrm {D} _ {\mathbb {X}}} \right\},
$$

Here, in case of a tie when choosing the  $k$ -th smallest, we break the tie randomly with uniform probability.

We have the following analogs of the convergence results (Theorems 7 and 8) for the  $k$ -nearest-neighbor estimator. The proofs are postponed to Section 4.4 and Section 4.5, respectively.

Theorem 10 Under Assumptions 2 and 3, and choosing  $k$  as

$$
k \sim \left\{ \begin{array}{l l} M ^ {\frac {2}{d _ {\mathbb {X}} + 2}}, & d _ {\mathbb {Y}} = 1, 2, \\ M ^ {\frac {d _ {\mathbb {Y}}}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}}, & d _ {\mathbb {Y}} \geq 3, \end{array} \right.
$$

there is a constant  $C > 0$  (which depends only on  $d_{\mathbb{X}}, d_{\mathbb{Y}}, L, \underline{c}$ ), such that, for all probability distribution  $\nu \in \mathcal{P}(\mathbb{X})$ , we have

$$
\mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \nu (\mathrm {d} x) \right] \leq \sup  _ {x \in \mathbb {X}} \mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \right] \leq C \times \left\{ \begin{array}{l l} M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}}, & d _ {\mathbb {Y}} = 1, \\ M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}} \ln M, & d _ {\mathbb {Y}} = 2, \\ M ^ {- \frac {1}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}}, & d _ {\mathbb {Y}} \geq 3. \end{array} \right. \tag {4}
$$

Theorem 11 Under Assumptions 3, for any  $\nu \in \mathcal{P}(\mathbb{X})$ , we have

$$
\operatorname {V a r} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \nu (\mathrm {d} x) \right] \leq \frac {1}{k}. \tag {5}
$$

Moreover, if  $\nu$  is dominated by  $\lambda_{\mathbb{X}}$  with a constant  $\overline{C} > 0$ , then

$$
\begin{array}{l} \mathrm {V a r} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \nu (\mathrm {d} x) \right] \\ \leq \frac {2 ^ {2 d _ {\mathbb {X}} + 1} \overline {{C}} ^ {2} M}{\underline {{c}} ^ {2} k ^ {2}} \left(\left(8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} + \frac {k}{M - 1}\right) ^ {2} + \frac {\sqrt {2 \pi}}{\sqrt {M - 1}} \left(8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} + \frac {k}{M - 1}\right) + \frac {4}{M - 1}\right). \\ \end{array}
$$

With  $k$  chosen as in Theorem 10, this reduces to

$$
\operatorname {V a r} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \nu (\mathrm {d} x) \right] \lesssim \left\{ \begin{array}{l l} M ^ {- \frac {2 (2 \vee d _ {\mathbb {Y}})}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}} \ln (M), & 2 \vee d _ {\mathbb {Y}} \leq d _ {\mathbb {X}}, \\ M ^ {- 1}, & 2 \vee d _ {\mathbb {Y}} > d _ {\mathbb {X}}. \end{array} \right.
$$

# 2.4 Comments on the convergence rate

This section gathers several comments on the convergence results we have developed in Sections 2.2 and 2.3.

# 2.4.1 ON THE CONVERGENCE RATE

We first comment on the expectations in Theorem 7 and 10.

Sharpness of the bounds. Currently, we cannot establish the sharpness of the convergence rates in Theorems 7 and 10. However, we can compare our results to established results in similar settings. For  $d_{\mathbb{Y}} = 1$ , we may compare it to the optimal rate of non-parametric regression of a Lipschitz continuous function. It is shown in Stone (1982) that the optimal rate is  $M^{-\frac{1}{d_{\mathbb{X}} + 2}}$ , the same as in Theorems 7 and 10 when  $d_{\mathbb{Y}} = 1$ . For  $d_{\mathbb{Y}} \geq 3$ , as noted in Backhoff et al. (2022), we may compare to the Wasserstein convergence rate of empirical measure in the estimation of a static distribution on  $\mathbb{R}^{d_{\mathbb{X}} + d_{\mathbb{Y}}}$ . We refer to Fournier and Guillin (2015) for the optimal rate, which coincides with those in Theorems 7 and 10.

Error components. We discuss the composition of our upper bound on the expected average Wasserstein error by dissecting the proof of Theorem 7 and 10. In the proofs, we decompose the expected average errors into two components: approximation error and estimation error. The approximation error occurs when treating  $P_{x'}$  as equal to  $P_x$  when  $x'$  is close to the query point  $x$ , leading to an error of size  $L\|x - x'\|_{\infty}$ . The estimation error is associated with the Wasserstein error of empirical measure under i.i.d. sampling (see (21)). From Definitions 5 and 9, the  $r$ -box estimator effectively manages the approximation error but struggles with controlling the estimation error, whereas the  $k$ -nearest-neighbor estimator exhibits the opposite behavior.

Explicit bounds. We primarily focus on analyzing the convergence rates of the  $r$ -box and  $k$ -nearest-neighbor estimators as  $M \to \infty$ . Therefore, within the proofs of these results, we track only the rates (and ignore various constant coefficients). If more explicit bounds are preferred, intermediate results such as (23), or (27) could be good starting points for computing them. Additionally, in Section A, we provide a numerical illustration that highlights the impact of the Lipschitz constant  $L$  and the choices of  $r$  and  $k$  on the bounds.

# 2.4.2 ON THE FLUCTUATION

We next discuss the variances studied in Theorems 8 and 11. In Appendix D, we also include results derived from the Azuma-Hoeffding inequality (e.g., (Wainwright, 2019, Corollary 2.20)), though they provide rougher rates.

Condition that  $\nu$  is dominated by  $\lambda_{\mathbb{X}}$ . In Theorems 8 and 11, we assume that the  $\nu$  is dominated by  $\lambda_{\mathbb{X}}$ . This assumption is somewhat necessary. To illustrate, let us examine the nonparametric regression problem under a comparable scenario. We consider a fixed query point. In this context, the central limit theorem for  $k$ -nearest-neighbor estimator is well-established, and the normalizing rate is  $k^{-\frac{1}{2}}$  (cf. (Biau and Devroye, 2015, Theorem 14.2)). This suggests that the rate in (5) is sharp. For the  $r$ -box estimator, we believe that a supporting example can be constructed where  $\nu$  is highly concentrated. On the other hand, we conjecture that if  $\xi \sim \nu$ , the variance could potentially attain the order of  $M^{-1}$ . For a pertinent result, we direct the reader to (Backhoff et al., 2022, Theorem 1.7).

Sharpness of the bounds. Regarding the variance in Theorem 8, it is upper bounded by the commonly observed order of  $M^{-1}$ . We believe that this rate is sharp, though we do not have a proof at this time. As for Theorem 11, the variance is subject to a rougher rate when  $2 \vee d_{\mathbb{Y}} \leq d_{\mathbb{X}}$ . We, however, conjecture that this variance attains the order of  $M^{-1}$  as long as  $\nu$  is dominated by  $\lambda_{\mathbb{X}}$ .

# 2.5 Towards implementation with neural networks

In light of recent practices in machine learning, during the learning of  $P$ , we may combine the  $r$ -box method or  $k$ -nearest-neighbor method into the training of certain parameterized model. To this end we let

$$
\tilde {P}: \mathbb {T} \times \mathbb {X} \to \mathcal {P} (\mathbb {Y})
$$

$$
(\theta , x) \mapsto \tilde {P} _ {x} ^ {\theta}
$$

be a parameterized model (e.g., a neural network), where  $\mathbb{T}$  is the parameter space and  $\theta \in \mathbb{T}$  is the parameter to be optimized over. Given an integer  $N \geq 1$ , we may train  $\tilde{P}^{\theta}$  on a set of query points  $\mathcal{Q} = (\tilde{X}_n)_{n=1}^N$  satisfying the assumption below.

Assumption 12 The query points  $\mathcal{Q} = \{(\tilde{X}_n)\}_{n=1}^N$  are i.i.d. with uniform distribution over  $\mathbb{X}$ , and are independent of the data points  $\mathcal{D} = \{(X_m, Y_m)\}_{m=1}^M$ .

We propose the training objectives below

$$
\underset {\theta \in \mathbb {T}} {\arg \min } \frac {1}{N} \sum_ {n = 1} ^ {N} \mathcal {W} \left(\hat {P} _ {\tilde {X} _ {n}} ^ {r}, \tilde {P} _ {\tilde {X} _ {n}} ^ {\theta}\right) \quad \text {o r} \quad \underset {\theta \in \mathbb {T}} {\arg \min } \frac {1}{N} \sum_ {n = 1} ^ {N} \mathcal {W} \left(\check {P} _ {\tilde {X} _ {n}} ^ {k}, \tilde {P} _ {\tilde {X} _ {n}} ^ {\theta}\right), \tag {6}
$$

that is, minimize the mean of 1-Wasserstein errors between the parametrized model and the empirical  $r$ -box (or  $k$ -nearest-neighbour) approximation of the conditional distribution at the location of the random query points.

The following proposition together with Theorem 7 or Theorem 10 justifies using the objectives in (6). It is valid for any estimator for  $P$  that satisfies the bounds in (3) or (4). Centered on appropriate Lipschitz continuity conditions, the proposition offers insights into the worst-case performance guarantees. The proof is deferred to Section 4.6. We also refer to Altekrüger et al. (2023) for a worst-case performance guarantee for conditional generative models, which is contingent upon Lipschitz continuity. For related practical approaches, we refer to, for example, Nguyen et al. (2024) and the references therein. In contrast, similar guarantees for the  $r$ -box and  $k$ -nearest-neighbor estimators are more elusive due to their inherently piece-wise constant nature.

Proposition 13 Suppose Assumptions 2, 3, and 12 hold. Let  $\overline{P}$  of  $P$  be an estimator constructed using the data points  $\mathcal{D}$  only. Consider a training procedure that produces a (random)  $\Theta = \Theta(\mathcal{D}, \mathcal{Q})$  satisfying

$$
\sup  _ {x, x ^ {\prime} \in \mathbb {X}} \frac {\mathcal {W} \left(\tilde {P} _ {x} ^ {\Theta} , \tilde {P} _ {x ^ {\prime}} ^ {\Theta}\right)}{\| x - x ^ {\prime} \| _ {\infty}} \leq L ^ {\Theta} \tag {7}
$$

for some (random)  $L^{\Theta} > 0$ . Then,

$$
\begin{array}{l} \mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \tilde {P} _ {x} ^ {\Theta}\right) \mathrm {d} x \right] \leq \mathbb {E} \left[ (L + L ^ {\Theta}) \mathcal {W} \left(\lambda_ {\mathbb {X}}, \frac {1}{N} \sum_ {n = 1} ^ {N} \delta_ {\tilde {X} _ {n}}\right) \right] \tag {8} \\ + \mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} (P _ {x}, \overline {{P}} _ {x}) \mathrm {d} x \right] + \mathbb {E} \left[ \frac {1}{N} \sum_ {n = 1} ^ {N} \mathcal {W} (\overline {{P}} _ {\tilde {X} _ {n}}, \tilde {P} _ {\tilde {X} _ {n}} ^ {\Theta}) \right]. \\ \end{array}
$$

Moreover, with probability 1,

$$
\sup  _ {x \in \mathbb {X}} \mathcal {W} \left(P _ {x}, \tilde {P} _ {x} ^ {\Theta}\right) \leq \left(d _ {\mathbb {X}} + 1\right) ^ {\frac {1}{d _ {\mathbb {X}} + 1}} \left(L + L ^ {\Theta}\right) ^ {\frac {d _ {\mathbb {X}}}{d _ {\mathbb {X}} + 1}} \left(\int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \tilde {P} _ {x} ^ {\Theta}\right) \mathrm {d} x\right) ^ {\frac {1}{d _ {\mathbb {X}} + 1}}. \tag {9}
$$

Remark 14 Assuming  $L^{\Theta} \leq \overline{L}$  for some (deterministic)  $\overline{L} > 0$ , by (9) and Jensen's inequality, we have

$$
\mathbb {E} \left[ \sup _ {x \in \mathbb {X}} \mathcal {W} \left(P _ {x}, \tilde {P} _ {x} ^ {\Theta}\right) \right] \leq (d _ {\mathbb {X}} + 1) ^ {\frac {1}{d _ {\mathbb {X}} + 1}} (L + \overline {{L}}) ^ {\frac {d _ {\mathbb {X}}}{d _ {\mathbb {X}} + 1}} \mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} (P _ {x}, \tilde {P} _ {x} ^ {\Theta}) \mathrm {d} x \right] ^ {\frac {1}{d _ {\mathbb {X}} + 1}}.
$$

This together with (8) provides a worst-case performance guarantee for  $\tilde{P}^{\Theta}$ .

Remark 15 Proposition 13 along with Remark 14 provides insights into the worst-case performance guarantees, but more analysis is needed. Specifically, understanding the magnitude of  $L^{\Theta}$  and  $\mathbb{E}\left[\frac{1}{N}\sum_{n = 1}^{N}\mathcal{W}(\overline{P}_{\tilde{X}_n},\tilde{P}_{\tilde{X}_n}^{\Theta})\right]$  requires deeper knowledge of the training processes for  $\tilde{P}^{\Theta}$ , which are currently not well understood in the extant literature. Alternatively, in the hypothetical case where  $\tilde{P}^{\Theta} = P$ ,  $L^{\Theta}$  would match  $L$  as specified in Assumption 2, and  $\mathbb{E}\left[\frac{1}{N}\sum_{n = 1}^{N}\mathcal{W}\left(\overline{P}_{\tilde{X}_n},\tilde{P}_{\tilde{X}_n}^{\Theta}\right)\right]$  would obey Theorem 7 or 10. However, practical applications must also consider the universal approximation capability of  $\tilde{P}^{\theta}$ . To the best of our knowledge, research on universal approximation with regularity constraints remains relatively limited. For a somewhat related study, we refer to Hong and Kratsios (2024) who explore the approximation of real-valued functions under Lipschitz continuity constraints.

# 3. Implementation with neural networks

Let  $\mathbb{X}$  and  $\mathbb{Y}$  be equipped with  $\| \cdot \| _1$ . Following the discussion in Section 2.5, we let  $\tilde{P}^{\theta}:\mathbb{X}\to \mathcal{P}(\mathbb{Y})$  be parameterized by a neural network and develop an algorithm that trains  $\tilde{P}^{\theta}$  based on  $k$ -nearest-neighbor estimator. The  $k$ -nearest-neighbor estimator  $\tilde{P}^{k}$  is preferred as  $\check{P}_x^k$  consistently outputs  $k$  atoms. This regularity greatly facilitates implementation. For instance, it enables the use of 3D tensors during Sinkhorn iterations to enhance execution speed (see Section 3.1.2 later). We refer also to the sparsity part of Section 5.2 for another component that necessitates the aforementioned regularity of  $\tilde{P}^k$ . These components would not be feasible with the  $r$ -box estimator  $\hat{P}^r$ , as  $\hat{P}_x^r$

produces an undetermined number of atoms. Furthermore, there is a concern that in some realizations,  $\hat{P}_x^r$  at certain  $x$  may contain too few data points, potentially leading  $\tilde{P}_x^\Theta$  to exhibit unrealistic concentration.

We next provide some motivation for this implementation. For clarity, we refer to the  $r$ -box estimator and the  $k$ -nearest-neighbor estimator as raw estimators. Additionally, we refer to  $\tilde{P}^{\Theta}$ , once trained, as the neural estimator. While raw estimators are adequate for estimating  $P$  on their own, they are piece-wise constant in  $x$  by design. On the other hand, a neural estimator is continuous in  $x$ . This continuity provides a performance guarantee in  $\sup \mathcal{W}$  distance, as outlined in Proposition 13 and the following remark. Moreover, the neural estimator inherently possesses gradient information. As discussed in the introduction, this feature renders the neural estimators useful in downstream contexts where gradient information is important, e.g., when performing model-based reinforcement learning.

We construct  $\tilde{P}^{\theta}$  such that it maps  $x\in \mathbb{X}$  to atoms in  $\mathbb{Y}$  with equal probabilities. For the related universal approximation theorems, we refer to Kratsios (2023); Acciaio et al. (2024). We represent these atoms with a vector with  $N_{\mathrm{atom}}$  entries denoted by  $y^\theta (x) = (y_1^\theta (x),\ldots ,y_{N_{\mathrm{atom}}}^\theta (x))\in \mathbb{Y}^{N_{\mathrm{atom}}}$  where  $N_{\mathrm{atom}}\in \mathbb{N}$  is chosen by the user. In our implementation, we set  $N_{\mathrm{atom}} = k$ . To be precise, we construct  $\tilde{P}^{\theta}$  such that

$$
\tilde {P} _ {x} ^ {\theta} = \frac {1}{N _ {\mathrm {a t o m}}} \sum_ {j = 1} ^ {N _ {\mathrm {a t o m}}} \delta_ {y _ {j} ^ {\theta} (x)}, \quad x \in \mathbb {N}. \tag {10}
$$

This is known as the Lagrangian discretization (see (Peyre and Cuturi, 2019, Section 9)). In Algorithm 1, we present a high level description of our implementation of training  $\tilde{P}^{\theta}$  based on the raw  $k$ -nearest-neighbor estimator.

Algorithm 1 Deep learning conditional distribution in conjunction with  $k$ -NN estimator  
Input: data  $\{(X_m,Y_m)\}_{m = 1}^M$  valued in  $\mathbb{R}^{d_{\mathbb{X}}}\times \mathbb{R}^{d_{\mathbb{Y}}}$  , neural estimator  $\tilde{P}^{\theta}$  represented by  $y^\theta (x)$  as elaborated in (10), parameters such as  $k,N_{\mathrm{atoms}},N_{\mathrm{batch}}\in \mathbb{N}_{+}$  , and learning rate  $\eta_{\theta}$    
Output: trained parameter  $\Theta$  for the neural estimator   
1: repeat   
2: for  $n = 1,\dots ,N_{\mathrm{batch}}$  do   
3: generate a query point  $\tilde{X}_n\sim \mathrm{Uniform}(\mathbb{X})$    
4: find the  $k$  nearest neighbors of  $\tilde{X}_n$  from data  $(X_{m})_{m = 1}^{M}$  and collect accordingly  $(\tilde{Y}_{n,i})_{i = 1}^{k}$    
5: end for   
6: compute with Sinkhorn algorithm (Y is equipped with  $\| \cdot \| _1$  1   
 $L[\theta ]\coloneqq \sum_{n = 1}^{N_{\mathrm{batch}}}\mathcal{W}\left(\frac{1}{k}\sum_{i = 1}^{k}\delta_{\tilde{Y}_{n,i}},\frac{1}{N_{\mathrm{atom}}}\sum_{j = 1}^{N_{\mathrm{atom}}}\delta_{y_j^\theta (\tilde{X}_n)}\right)$  (11)   
7: update  $\theta \gets \theta -\eta_{\theta}\nabla_{\theta}L[\theta ]$    
8: until Convergence   
9: return  $\Theta = \theta$

# 3.1 Overview of key components

In this section, we outline the three key components of our implementation. Each of these components addresses a specific issue:

- Managing the computational cost arising from the nearest neighbors search.  
$\circ$  Implementing gradient descent after computing  $\mathcal{W}$

![](images/4eb957e305a3677a621da0614fbe43fb9d3153210a24f0ea4640bef41065f2c0.jpg)  
Figure 1: An instance of RBSP in  $[0,1]^2$ .

The 2D unit box is partitioned into 16 rectangles based on 500 samples from Uniform([0, 1]). Note that the overlap between the bounding rectangles is intentionally maintained. Each partitioning is performed along an axis selected at random, dividing the samples within the pre-partitioned rectangle according to a random ratio drawn from Uniform([0.45, 0.55]). The edge ratio for mandatory bisecting along the longest edge is 5. If this ratio is exceeded, partitioning along the longest edge is enforced. The black dots represent samples within the respective rectangle.

- Selecting an appropriate Lipschitz constant for the neural estimator, preferably at a local level.

Further details and ablation analysis on these three components can be found in Section 5.

# 3.1.1 APPROXIMATE NEAREST NEIGHBORS SEARCH WITH RANDOM BINARY SPACE PARTITIONING (ANNS-RBSP)

Given a query point, performing an exact search for its  $k$ -nearest-neighbor requires  $O(M)$  operations. While a single search is not overly demanding, executing multiple searches as outlined in Algorithm 1 can result in significant computational time, even when leveraging GPU-accelerated parallel computing. To address this, we use ANNS-RBSP as a more cost-effective alternative. Prior to searching, we sort  $(X_{m})_{m = 1}^{M}$  along each axis and record the order of indices. During the search, the data is divided into smaller subsets by repeatedly applying bisection on these sorted indices, with a random bisecting ratio, on a randomly chosen axis. Furthermore, we apply a restriction that mandates bisection along the longest edge of a rectangle when the edge ratio exceeds certain value (a hyper-parameter of the model). We record the bounding rectangle for each subset created through this partitioning process. Once partitioning is complete, we generate a small batch of query points within each rectangle and identify the  $k$  nearest neighbors for each query point within that same rectangle. For a visual representation of ANNS-BSP, we refer to Figure 1. Leveraging the sorted indices, we can reapply this partitioning method during every training episode without much computational cost. We refer to Section 5.1 for additional details. There are similar ideas in the extant literature (cf. Hajebi et al. (2011); Ram and Sinha (2019); Li et al. (2020a)). Given the substantial differences in our setting, however, we conduct further empirical analysis in Section 5.1 to showcase the advantage of our approach against exact search.

# 3.1.2 COMPUTING  $\mathcal{W}$  FOR GRADIENT DESCENT

The following discussion pertains to the computation of (11), with the subsequent gradient descent in consideration. For simplicity, let us focus on the summand and reduce the problem to the following minimization. Let  $(\tilde{y}_1,\dots ,\tilde{y}_k)\in \mathbb{Y}^k$  be fixed, we aim to find

$$
\underset {y \in \mathbb {Y} ^ {n}} {\arg \min } \mathcal {W} \left(\frac {1}{k} \sum_ {i = 1} ^ {k} \delta_ {\tilde {y} _ {i}}, \frac {1}{n} \sum_ {j = 1} ^ {n} \delta_ {y _ {j}}\right). \tag {12}
$$

The criterion in (12) is convex as  $\mathcal{W}$  is convex in both arguments (cf. (Villani, 2008, Theorem 4.8)). To solve (12), as is standard, we cast it into a discrete optimal transport problem. To do so, first introduce the  $(k\times n)$ -cost matrix  $\mathsf{C}_y$ , where  $\mathsf{C}_{y,ij} \coloneqq \| \tilde{y}_i - y_j\|_1$ . As the criterion in (12) has uniform weights on the atoms, we next aim to solve the problem

$$
\underset {\mathsf {T} \in [ 0, 1 ] ^ {k \times n}} {\arg \min } \left\{\varphi_ {y} (\mathsf {T}) := \sum_ {(i, j) \in \{1, \dots , k \} \times \{1, \dots , n \}} \mathsf {T} _ {i j} \mathsf {C} _ {y, i j} \right\} \tag {13}
$$

$$
\mathrm {s u b j e c t t o} \quad \sum_ {j = 1} ^ {n} \mathsf {T} _ {i j} = \frac {1}{k}, i = 1, \ldots , k \quad \mathrm {a n d} \quad \sum_ {i = 1} ^ {k} \mathsf {T} _ {i j} = \frac {1}{n}, j = 1, \ldots , n.
$$

Let  $\mathsf{T}_y^*$  be an optimal transport plan that solves (13) for  $y$  fixed. Taking derivative of  $y\mapsto \varphi_y(\cdot)$  yields

$$
\left. \partial_ {y _ {j}} \varphi_ {y} (\mathsf {T}) \right| _ {\mathsf {T} = \mathsf {T} _ {y} ^ {*}} = \sum_ {i \in \{1, \dots , k \}} \mathsf {T} _ {y, i j} ^ {*} \partial_ {y _ {j}} \| \tilde {y} _ {i} - y _ {j} \| _ {1}, \quad j = 1, \dots , n. \tag {14}
$$

This gradient is in general not the gradient corresponding to (12), as  $\mathsf{T}_y^*$  depends on  $y$ , while (14) excludes such dependence. Nevertheless, it is still viable to update  $y$  using the gradient descent that employs the partial gradient specified in (14). To justify this update rule, first consider  $y' \in \mathbb{Y}$  satisfying  $\varphi_{y'}(\mathsf{T}_y^*) \leq \varphi_y(\mathsf{T}_y^*)$ , then observe that

$$
\mathcal {W} \left(\frac {1}{k} \sum_ {i = 1} ^ {k} \delta_ {\tilde {y} _ {i}}, \frac {1}{n} \sum_ {j = 1} ^ {n} \delta_ {y _ {j} ^ {\prime}}\right) \leq \varphi_ {y ^ {\prime}} (\mathsf {T} _ {y} ^ {*}) \leq \varphi_ {y} (\mathsf {T} _ {y} ^ {*}) = \mathcal {W} \left(\frac {1}{k} \sum_ {i = 1} ^ {k} \delta_ {\tilde {y} _ {i}}, \frac {1}{n} \sum_ {j = 1} ^ {n} \delta_ {y _ {j}}\right).
$$

This inequality is strict if  $\varphi_{y'}(\mathsf{T}_y^*) < \varphi_y(\mathsf{T}_y^*)$ . We refer to (Peyre and Cuturi, 2019, Section 9.1) and the reference therein for related discussions.

The Sinkhorn algorithm, which adds an entropy regularization, is a widely-used algorithm for approximating the solution to (13). Specifically, here, it is an iterative scheme that approximately solves the following regularized problem, subject to the constraints in (13),

$$
\underset {T ^ {\epsilon} \in [ 0, 1 ] ^ {k \times n}} {\arg \min } \left\{\sum_ {i, j \in \{1, \dots , k \} \times \{i, \dots , n \}} T _ {i j} ^ {\epsilon} C _ {i j} + \epsilon \sum_ {i, j \in \{1, \dots , k \} \times \{i, \dots , n \}} T _ {i j} ^ {\epsilon} (\log T _ {i j} - 1) \right\}, \tag {15}
$$

where  $\epsilon > 0$  is a hyper-parameter, and should not be confused with the  $\varepsilon$  used elsewhere. We refer to Section 5.2 for further details. We also refer to (Peyre and Cuturi, 2019, Section 4) and the reference therein for convergence analysis of the Sinkhorn algorithm. It is well known that the regularization term in (15) is related to the entropy of a discrete random variable. Larger values of  $\epsilon$  encourages the regularized optimal transport plan to be more diffusive. That is, for larger values

of  $\epsilon$ , the mass from each  $y_{j}$  is distributed more evenly across all  $\tilde{y}_i$ 's. Performing gradient descent along the direction in (14) tends to pull  $y_{j}$ 's towards the median of the  $\tilde{y}_i$ 's, as we are equipping  $\mathbb{Y}$  with the norm  $\| \cdot \|_1$ . Conversely, small values of  $\epsilon$  often leads to instability, resulting in NaN loss/gradient. To help with these issues, we implement the Sinkhorn algorithm after normalizing the cost matrix. Additionally, we use a large  $\epsilon$  (e.g., 1) in the first few training episodes, then switch to a smaller  $\epsilon$  (e.g., 0.1) in later episodes. Furthermore, we impose sparsity on the transport plan by manually setting the smaller entries of the transport plan to 0. The specific detailed configurations and related ablation analysis are provided in Section 5.2 and Appendix C.

# 3.1.3 NETWORK STRUCTURE THAT INDUCES LOCALLY ADAPTIVE LIPSCHITZ CONTINUITY

As previously discussed, it is desirable for the neural estimator to exhibit certain Lipschitz continuity. In practice, however, determining an appropriate Lipschitz constant for training the neural estimator  $\tilde{P}^{\theta}$  is challenging, largely because understanding the true Lipschitz continuity of  $P$  (if it exists) in a data-driven manner is very challenging. Additionally, the estimate provided in Proposition 13 is probabilistic. Fortunately, a specific network structure allows the neural estimator, when properly trained, to exhibit locally adaptive Lipschitz continuity. Subsequently, we provide a high-level overview of this network structure. Further detailed configurations and ablation analysis are presented in Section 5.3 and Appendix C.

Consider a fully connected feed-forward neural network with equal width hidden layers and layer-wise residual connection (He et al. (2016)). Let  $N_{\mathrm{neuron}}$  denote the width of the hidden layers. For activation, we use Exponential Linear Unit (ELU) function (Clevert et al. (2016)), denoted by  $\sigma$ . For hidden layers, we employ the convex potential layer introduced in Meunier et al. (2022),

$$
\mathsf {x} _ {\mathrm {o u t}} = \mathsf {x} _ {\mathrm {i n}} - \| \mathsf {W} \| _ {2} ^ {- 1} \mathsf {W} ^ {\mathsf {T}} \sigma (\mathsf {W} \mathsf {x} _ {\mathrm {i n}} + \mathsf {b}). \tag {16}
$$

By (Meunier et al., 2022, Proposition 3), the convex potential layer is 1-Lipschitz continuous in  $\| \cdot \|_2$  sense. For the input layer, with a slight abuse of notation, we use

$$
\mathsf {x} _ {\text {o u t}} = N _ {\text {n e u r o n}} ^ {- 1} \operatorname {d i a g} \left(\left| \mathsf {W} \right| _ {1} ^ {- 1} \wedge 1\right) \sigma \left(\mathsf {W} \mathsf {x} _ {\text {i n}} + \mathsf {b}\right), \tag {17}
$$

where  $|W|_1$  computes the absolute sum of each row of the weight matrix to form a vector of size  $N_{\mathrm{neuron}}$ , the reciprocal and  $\cdot \wedge 1$  are applied entry-wise, and diag produces a diagonal square matrix based on the input vector. In short, the normalization in (17) is only applied to the rows of  $W$  with  $\ell_1$ -norm exceeding 1. Consequently, the input layer is 1-Lipschitz continuous in  $\| \cdot \|_1$  sense. A similar treatment is used for the output layer but without activation,

$$
\mathrm {x} _ {\text {o u t}} = L d _ {\mathbb {Y}} ^ {- 1} \operatorname {d i a g} \left(\left| \mathrm {W} \right| _ {1} ^ {- 1} \wedge 1\right) \left(\mathrm {W x} _ {\text {i n}} + \mathrm {b}\right). \tag {18}
$$

where  $L > 0$  is a hyper-parameter. The output represents atoms on  $\mathbb{Y}$  with uniform weight, therefore, no  $N_{\mathrm{atom}}^{-1}$  is required here.

The spectral norm  $\| \mathsf{W}\| _2$  in (16), however, does not, in general, have an explicit expression. Following the implementation in Meunier et al. (2022), we approximate each  $\| \mathsf{W}\| _2$  with power iteration. Power iterations are applied to all hidden layers simultaneously during training. To control the pace of iterations, we combine them with momentum-based updating. We refer to Algorithm 2 for the detailed implementation. Our implementation differs from that in Meunier et al. (2022), as Meunier et al. (2022) controls the frequency of updates but not the momentum. In a similar manner, for input and output layers, instead of calculating the row-wise  $\ell_1$ -norm explicitly, we update them with the same momentum used in the hidden layers. Our numerical experiments consistently show that a small momentum value of  $\tau = 10^{-3}$  effectively maintains

Algorithm 2 Power iteration with momentum for updating  $\| W\|_2$  estimate, applied to all convex potential layers simultaneously at every epoch during training

Input: weight matrix  $\mathsf{W} \in \mathbb{R}^{d \times d}$  of a convex potential layer, previous estimate  $\hat{h} \in \mathbb{R}$  and auxiliary vector  $\hat{\mathbf{u}} \in \mathbb{R}^d$ , momentum  $\tau \in (0,1)$

Output: updated  $\hat{h}$  and  $\hat{\mathbf{u}}$ , in particular,  $\hat{h}$  will be used as a substitute of  $\| W \|_2$  in (16)

1:  $\mathbf{v} \gets \mathbf{W}\hat{\mathbf{u}} / \| \mathbf{W}\hat{\mathbf{u}}\|_2$  
2:  $\mathsf{u}\gets \mathsf{W}^{\mathrm{T}}\mathsf{v} / \| \mathsf{W}^{\mathrm{T}}\mathsf{v}\|_{2}$  
3:  $h \gets 2 / \left( \sum_{i} (\mathsf{W} \mathsf{u} \cdot \mathsf{v})_i \right)^2$  
4:  $\hat{h} \gets \tau \hat{h} + (1 - \tau) h$  
5:  $\hat{\mathbf{u}}\gets \tau \hat{\mathbf{u}} +(1 - \tau)\mathbf{u}$  
6: return  $\hat{h}, \hat{\mathbf{u}}$

power  
teration  
mentum-

based

updating

adaptive continuity while maintaining a satisfactory accuracy. The impact of  $L$  in (18) and  $\tau$  in Algorithm 2 is discussed in Section 5.3.

During training, due to the nature of our updating schemes, the normalizing constants do not achieve the values required for the layers to be 1-Lipschitz continuous. We hypothesize that this phenomenon leads to a balance that ultimately contributes to adaptive continuity: on one hand, the weights  $W$  stretch to fit (or overfit) the data, while on the other, normalization through iterative methods prevents the network from excessive oscillation. As shown in Section 5.3.2 and 5.3.3, the  $L$  value in (18) and the momentum  $\tau$  in Algorithm 2 affect the performance significantly. For completeness, we also experiment with replacing (16) by fully connected feedforward layers similar to (17), with or without batch normalization (Ioffe and Szegedy (2015)) after affine transformation. This alternative, however, failed to produce satisfactory results.

# 3.2 Experiments with synthetic data

We consider data simulated from three different models. The first two have  $d_{\mathbb{X}} = d_{\mathbb{Y}} = 1$ , while the third has  $d_{\mathbb{X}} = d_{\mathbb{Y}} = 3$ . Here we no longer restrict  $\mathbb{Y}$  to be the unit box, however, we still consider  $\mathbb{X}$  to be a  $d_{\mathbb{X}}$ -dimensional unit box (not necessarily centered at the origin).

In Model 1 and 2,  $X \sim \mathrm{Uniform}([0,1])$ . Model 1 is a mixture of two independent Gaussian random variables with mean and variance depending on  $x$ ,

$$
Y = \xi \Big (0. 1 \left(1 + \cos (2 \pi X)\right) + 0. 1 2 \left| 1 - \cos (2 \pi X) \right| Z + 0. 5 \Big),
$$

where  $Z \sim \mathrm{Normal}(0,1)$  and  $\xi$  is a Rademacher random variable independent of  $Z$ . For Model 2, we have

$$
Y = 0. 5 \mathbb {1} _ {[ 0, 1)} (X) + 0. 5 U,
$$

where  $U \sim \mathrm{Uniform}([0,1])$ . The conditional distribution in Model 2 is intentionally designed to be discontinuous in the feature space. This choice was made to evaluate performance in the absence of the Lipschitz continuity stipulated in Assumption 2. Model 3 is also a mixture of two independent Gaussian random variables, constructed by considering  $X \sim \mathrm{Uniform}([- \frac{1}{2}, \frac{1}{2}]^3)$  and treating  $X$  as a column vector (i.e.,  $X$  take values in  $\mathbb{R}^{3 \times 1}$ ),

$$
Y = \zeta \Big (\cos (\mathsf {A} X) + 0. 1 \cos (\Sigma_ {X}) W \Big) + (1 - \zeta) \Big (\cos (\mathsf {A} ^ {\prime} X) + 0. 1 \cos (\Sigma_ {X} ^ {\prime}) W ^ {\prime} \Big).
$$

Above, the cos functions act on vector/matrix entrywise,  $\mathsf{A} \in \mathbb{R}^{3 \times 3}$ , and  $\Sigma_x$  also takes value in  $\mathbb{R}^{3 \times 3}$ . Each element of  $\Sigma_x$  is defined as  $\mathsf{v}_{ij}x$  for some  $\mathsf{v}_{ij} \in \mathbb{R}^{1 \times 3}$ . The entries of  $\mathsf{A}$  and  $\mathsf{v}_{ij}$  are drawn from

standard normal in advance and remain fixed throughout the experiment. The matrices  $\mathsf{A}'$  and  $\Sigma_x'$  are similarly constructed. Furthermore,  $W$  and  $W'$  are independent three-dimensional standard normal r.v.s, while  $\zeta$  represents the toss of a fair coin, independent of  $X$ ,  $W$ , and  $W'$ .

For the purpose of comparison, two different network structures are examined. The first, termed LipNet, is illustrated in Section 3.1. The second, termed StdNet, is a fully connected feedforward network with layer-wise residual connections He et al. (2016), ReLU activation, and batch normalization immediately following each affine transformation, without specifically targeting Lipschitz continuity. With a hyper-parameter  $k$  for the  $k$ -nearest-neighbor estimator, which we specify later, each network contains 5 hidden layers with  $2k$  neurons. These networks are trained using the Adam optimizer Kingma and Ba (2017) with a learning rate of  $10^{-3}$ . For StdNet in Model 1 and 2, the learning rate is set to 0.01, as it leads to better performance. Other than the learning rates, StdNet and LipNet are trained with identical hyper-parameters across all models. We refer to Appendix C for a summary of hyper-parameters involved.

We generate  $10^{4}$  samples for Models 1 and 2. Given the convergence rate specified in Theorem 10, we note that the sample size are considered relatively small. For these two models, we chose  $k = 100$  and utilized neural networks  $\tilde{P}^{\theta}$  that output atoms of size  $N_{\mathrm{atom}} = k$ . The choice of  $k$  is determined by a rule of thumb. In particular, our considerations include the magnitude of  $k$  suggested by Theorem 10 and the computational costs associated with the Sinkhorn iterations discussed in Section 3.1.2. The results under Model 1 and 2 are plotted in Figure 2, 3 and 4. Figure 2 provides a perspective on joint distributions, while Figure 3 and 4 focus on conditional CDFs across different  $x$  values.

Figure 2 suggests that both StdNet and LipNet adequately recover the joint distribution. The LipNet's accuracy is, however, notably superior and produces smooth movements of atoms (as seen in the third row of Figure 2). Although further fine-tuning may provide slight improvements in StdNet's performance, StdNet will still not achieve the level of accuracy and smoothness observed in LipNet. The average absolute value of derivative of each atom (fourth row of Figure 2), makes it evident that LipNet demonstrates a capacity of automatically adapting to a suitable level of Lipschitz continuity locally. In particular, in Model 2, the atoms of LipNet respond promptly to jumps while remaining relatively stationary around values of  $x$  where the kernel is constant. We emphasize that LipNet is trained using the same hyper-parameters across Models 1, 2, and 3.

Figure 3 shows the estimated conditional distribution at different values of  $x$ . Figure 3 indicates that the raw  $k$ -nearest-neighbor estimator deviates frequently from the actual CDFs. This deviation of the raw  $k$ -nearest-neighbor estimator is expected, as it attempts to estimate an unknown CDF with only  $k = 100$  samples given an  $x$ . Conversely, the neural estimator, especially the LipNet, appears to offer extra corrections even if they are trained based on the raw  $k$ -nearest-neighbor estimator. This could be attributed to neural estimators implicitly leveraging information beyond the immediate neighborhood.

Figure 4 compares the  $\mathcal{W}$ -distance between each estimator and the true conditional distribution at various values of  $x$ , using the following formula (see (Peyre and Cuturi, 2019, Remark 2.28)),

$$
\mathcal {W} (F, G) = \int_ {\mathbb {R}} | F (r) - G (r) | \mathrm {d} r, \tag {19}
$$

where  $F$  and  $G$  are CDFs. This quantity can be accurately approximated with trapezoidal rule. In Model 1, the neural estimator generally outperforms the raw estimator with  $k = 100$  across most values of  $x$ , even though the raw estimator is used for training the neural estimators. Furthermore, LipNet continues to outperform raw estimators with larger values of  $k$  - even though LipNet is trained with a raw estimator with  $k = 100$ . In Model 2, LipNet continues to demonstrate a superior performance, except when compared to the raw estimator with  $k = 1,000$  at  $x$  distant from 0.5,

![](images/793a1960c8dd8239a88a4b80d98a52be9a77a269322214b761f74e1a4e3cdfdd.jpg)

![](images/daed212e20113354d216e54b9f05e1b41eb0aaf862d720ac991daa40aa55a457.jpg)

![](images/130a9cee93da502383ed315ff82115d69056894a5fcf7d9c1100fe36c886ec61.jpg)

![](images/fb8968bf729392ba2fac9ba1e27a2cdc358b31317971ada3c9a507d1340f5d31.jpg)

![](images/e8c82fa00ce2d9bc56b7336decb3b096fba4e7d03d4ccc5f966b8fa23768c8ab.jpg)

![](images/133565c08a54b77e426ac59a4d8d1f87bef0a8756d87c8ec42e54399c8d3d61b.jpg)

![](images/45911e26d238b82298041e457cf41cbc6e94fa531d577abdb7d484c0f65737e9.jpg)

![](images/616db4f093b10ff52c075700dfdddfc276822a8f04ad8ba0fa0c81c4db83529e.jpg)

![](images/abc5766787089c637c25fda0ad752226541223129a3c29414a2bed100f0dee8a.jpg)

![](images/4bc1f9af1553dd5ff913f4b29ef6fd801f6b2644e5a50e8759dd947d719147b2.jpg)

![](images/1f96bcacca4a8a7c2011bf8ff6d5bba78e2fc19039c0e67fa0624c4db409e6d1.jpg)

![](images/d3c713c23069b302420bce6668eeab9c4571597dd288e1a93a5ecdd74e4c00b5.jpg)

![](images/01ec52554f85860311f8a6cd6b6bed46bc08d75a435bef2af6b2a4faa77cabbd.jpg)  
(a) Model 1, StdNet

![](images/5a7f08223a595b59cfbd0f773c85b29285b2b8d88a573c83198ecaf749e39ea4.jpg)  
(b) Model 1, LipNet

![](images/23995073a9507a896732d456fe5208617377137fa64d7285599d36e61d62fede.jpg)  
(c) Model 2, StdNet

![](images/b9a5e20ff6bbdfe5d57a4973daed15378d1466f355d9e14fb5bc6ae2f761404d.jpg)  
Figure 2: Various estimators under Model 1 and 2, joint distributions.  
(d) Model 2, LipNet

The first row presents the data. Neural networks with different structures are trained on the same set of data for comparison. The second row shows scatter plots of atoms at various  $x$  values. The third row illustrates the evolution of 20 atoms as  $x$  varies. The final row presents the average of the derivative of each atom with respect to  $x$ , with a notable difference in the  $y$  axis scale.

![](images/7c954ded12b6928489873e16413588d21f7e94b6e26699fec723a4461b0f6ebb.jpg)

![](images/0ce34eed4276da7311102aae87f0cf388e38083a457cad5498cf40df827d6ed9.jpg)

![](images/63e264d5d05828b6891ae3cf677d792550005cdb4c994db73b00ba370a4bbad3.jpg)

![](images/67ee21ee7a02448b25d887e48f853b1f5900ef548742efcaf39a8f5d4adc132b.jpg)

![](images/ab223e97a0ccead7c0af7b45b159a2b917c7b3c51c4fb2feffd0d2a8caa6493f.jpg)

![](images/729bf3ef80013b8f3e28e912395d7827ecfe7aba894ca33f567149be99c5327c.jpg)

![](images/f954c03a6dcdc480047e47b5157a0291734e3824843e33375c5bb2065d5f466a.jpg)

![](images/3d522b945643c30d26920c30c22e530600fdaa42742ce5010a0a2e3030855c6e.jpg)

![](images/eb35fa7880f02dec77356956386df938349a67efbea225d309745aa0ea12b88b.jpg)  
Figure 3: Various estimators under Model 1 and 2, conditional CDFs.

We compare the conditional CDFs at various values of  $x$ , derived from StdNet, LipNet, the raw  $k$ -nearest estimator with  $k = 100$  (also used in the training of StdNet and LipNet), and the ground truth. The first row pertains to data set 1. The second row pertains to data set 2. Subfigure titles display the values of  $x$ .

![](images/bcaeeca04e1fa4c9145b71942ce76aa8fcd83dd7d84768a6e86a1514329a51e5.jpg)

![](images/19e106673d5c810d9831af4daa96135c38802a4eed37664837b9a17a00578d99.jpg)  
Figure 4: Errors at different  $x$ 's of various estimators under Model 1 and 2.

We compute the  $\mathcal{W}$ -distance between estimators and the true conditional distribution at different  $x$ 's. StdNet and LipNet are trained based on the raw  $k$ -nearest-neighbor estimator with  $k = 100$ .

the reason is that, here, the conditional distribution is piece-wise constant in  $x$ , which enhances the performance of the raw estimator at larger  $k$  values.

The aforementioned findings indicate superior performance by LipNet. We, however, recognize that improvements are not always guaranteed, as demonstrated in Figures 3 and 4.

![](images/a380fcd06aa8589b1e2f24625d0d819c53f08e199752e5861817cab8840bee70.jpg)

![](images/069caaee607f4e1f3f541171e0259236e49fa0ecee51064a9f28f0699dc6b6b8.jpg)

![](images/826263135cd256cfddf593d706f999883fcbf07df616089f021bbfc3e0862ba4.jpg)

![](images/48a128e400e44871b8417bf8984208fd5a64405f66ad4008f198d54aa4d19189.jpg)

![](images/8bb12d9f53964d8185c11b07c01440d76cdced8e60e0d788af6a7947009ba230.jpg)

![](images/b661a4cdd4e74704872c271d0ea97d1596baf5f2612238a4ea7984dcbe24d425.jpg)

![](images/2cb3549e2c278eee95b0b62e404f94d94b7acaa8cca28e2c787a6f9f21e38e9f.jpg)

![](images/f6985c7021da55518a830f6f1205bafa6543848d90384cb70151111c0ddbe5d5.jpg)

![](images/baa87650345b5246186bd45199dff69e01699bc96b4f16ce58ca9ad6d839f749.jpg)  
Figure 5: Various estimators under Model 3, projections of conditional CDFs.

![](images/bc01ee422c42d2a312a24c185716dde8d663faa7e03cd428e80cb70437f93980.jpg)

![](images/28ca0c922c1e6442ba010af18ea972626a7aed992bca3644dc27fa5cd0ff9334.jpg)

![](images/539d33a833689e7641f9c5a456a39842d216c299568d6fa2b25f68b370b17493.jpg)

![](images/4b949f156ddb3e894d6935506e9bd51bef35873975d939edc5d2c2b8a9399022.jpg)

We compare the projected conditional CDFs at  $x = (0.12, -0.33, 0.1)$ . The estimations are obtained from StdNet, LipNet, the raw estimator with  $k = 300$  (also used in training StdNet and LipNet), and the ground truth. Subfigure titles display the vectors used for projection. Note the difference in the  $x$  axis scale.

For Model 3, we generate  $10^{6}$  samples and select  $k = 300$ . We train both neural estimators using Adam optimizer with a learning rate of  $10^{-3}$ . Hyperparameters such as  $L$  in (18) and  $\tau$  in Algorithm 2 are consistent with those used for Models 1 and 2. We refer to Table 4 for the detailed configuration.

In Figure 5, we visualize the outcomes in Model 3: the conditional CDFs at an arbitrarily chosen  $x$  are projected onto various vectors. We observe that the neural estimators considerably outperform the raw  $k$ -nearest-neighbor estimator, likely owing due to their implicit use of global information outside of the immediate neighbors during training. For further comparisons, we present additional figures in Appendix B: Figures 15, 16 and 17 feature the exact same neural estimators as shown in Figure 5, but with the raw  $k$ -nearest-neighbor estimators employing different  $k$  values,  $k = 1,000, 3,000, 10,000$ . Raw  $k$ -nearest-neighbor estimators with  $k = 1,000, 3,000$  are superior to that with  $k = 300$ , while at  $k = 10,000$ , the accuracy begins to decline. Upon comparison,

![](images/efe29a6554d2444c07748cb3bc4aba3f0171b03a583e4b6e4775869bb09aba01.jpg)  
Raw  $k = 1,000$

![](images/a5a4498b3146739b730262ba92fb235a83ddb635232fa17ae6c8369c41a3faa8.jpg)  
Raw  $k = 3,000$

![](images/07a8fa09f7d0920182e26301faef98e3188a83c8fd58a49b6dc7f0929b355a04.jpg)  
Figure 6: Histogram of 10,000 projected Wasserstein-1 errors.  
StdNet

![](images/65d3d0ffc9fde3477f0557569298628d19a03708a3cde01b2f30549eb81bfb09.jpg)  
LipNet

Each histogram consists of 20 uniformly positioned bins between 0 to 0.1. The errors of different estimators are computed with the same set of query points and projection vectors. Errors larger than 0.1 will be placed in the right-most bins. Note StdNet and LipNet are trained with  $k = 300$ .

the neural estimator trained with  $k = 300$  consistently outperforms the raw  $k$ -nearest-neighbor estimators for all values of  $k$ .

![](images/923b30cb27b8870e358da5ed3e25361a32bb7ab0735b4ac352601bb829b909d9.jpg)

![](images/b5c0774d463e27b0c32c7e2a7e81155046e94665068be49039feec0635f7c958.jpg)

![](images/2f8d01b371cbd6f91393d2db86d6d8e5dc0e07147f2fe4eb8be01fb40b47f5f2.jpg)

![](images/51ea61200702ac35bcd6c383d900dc616ab4db9285080b7cfb2457e26c952e76.jpg)  
[1,0,0]  
Figure 7: LipNet under Model 3, projected trajectories of 20 atoms.  
We illustrate the projected trajectories of 20 atoms by evaluating LipNet at 100 evenly allocated points along the straight line that intersects the origin and  $x = (0.12, -0.33, 0.1)$ , situated within  $[0, 1]^3$ . The  $x$ -axis denotes the specific points along the line, consistent across all subfigures. Subfigure titles display the vectors used for projection. Note the difference in the  $y$  axis scale.

![](images/7218d7a2ed09872f9ee25ce626dc691121e8058b70211001551355c978a4e569.jpg)  
[0,1,0]

![](images/8f81df6cf1445f3383fc3d25e2594051b58914cf9caa68c5d3ade347d10f0dbe.jpg)  
[0, 0, 1]

For a more comprehensive comparison, we randomly select 10,000 query points. For each query point, we randomly generate a vector in  $\mathbb{R}^3$ , normalized under  $\| \cdot \|_1$ , and project the atoms produced by the estimators onto said vector. With the same vector, we also compute the corresponding true CDFs of the projected  $Y$  given the query point. We then approximately compute the  $\mathcal{W}$ -distance

between the projected distributions via (19). The resulting histograms are shown in Figure 6, which suggests that LipNet performs best. The rationale for employing this projection approach, rather than directly computing the  $\mathcal{W}$ -distance between discrete and continuous distributions over  $\mathbb{R}^3$ , is due to the higher cost and lower accuracy of the latter approach (see also the discussion in Section 5.2). While this projection approach provides a cost-effective alternative for performance evaluation, it may not fully capture the differences between the estimations and ground truth.

Lastly, to demonstrate how atoms, in the neural estimator, move as  $x$  varies, Figure 7 shows the projected trajectories along a randomly selected straight line through the origin. The movement of atoms in LipNet is smooth, consistent with previous observations. Interestingly, the movement of atoms in StdNet isn't excessively oscillatory either, although its continuity is slightly rougher compared to LipNet. The reader may execute the Jupyter notebook on our github repository https://github.com/zcheng-a/LCD_kNN to explore the projected conditional CDFs and atoms' trajectories for different  $x$  values.

# 4. Proofs

# 4.1 Auxiliary notations and lemmas

In this section, we will introduce a few technical results that will be used in the subsequent proofs. We first define

$$
R (m) := \sup  _ {x \in \mathbb {X}} \int_ {\mathbb {Y} ^ {m}} \mathcal {W} \left(P _ {x}, \frac {1}{m} \sum_ {\ell = 1} ^ {m} \delta_ {y _ {\ell}}\right) \bigotimes_ {\ell = 1} ^ {m} P _ {x} (\mathrm {d} y _ {\ell}), \quad m \in \mathbb {N}. \tag {20}
$$

We stipulate that  $R(0) = 1$ . By Fournier and Guillin (2015), we have

$$
R (m) \leq \widehat {C} \times \left\{ \begin{array}{l l} m ^ {- \frac {1}{2}}, & d _ {\mathbb {Y}} = 1, \\ m ^ {- \frac {1}{2}} \ln (m), & d _ {\mathbb {Y}} = 2, \\ m ^ {- \frac {1}{d _ {\mathbb {Y}}}}, & d _ {\mathbb {Y}} \geq 3, \end{array} \right. \tag {21}
$$

for some constant  $\widehat{C} > 0$  depending only on  $d_{\mathbb{Y}}$ . For comprehension, we also point to Kloeckner (2020); Fournier (2023) for results that are potentially useful in analyzing explicit constant, though it is out of the scope of this paper.

The lemma below pertains to the so-called approximation error, which arises when treating data points  $Y_{j}$  with  $X_{j}$  around an query point as though they are generated from the conditional distribution at the query point.

Lemma 16 Under Assumption 2, for any integer  $J \geq 1$  and  $x, x_1, \ldots, x_J \in \mathbb{X}^{J+1}$ , we have

$$
\left| \int_ {\mathbb {Y} ^ {J}} \mathcal {W} \left(\frac {1}{J} \sum_ {j = 1} ^ {J} \delta_ {y _ {j}}, P _ {x}\right) \bigotimes_ {j = 1} ^ {J} P _ {x _ {j}} (\mathrm {d} y _ {j}) - \int_ {\mathbb {Y} ^ {J}} \mathcal {W} \left(\frac {1}{J} \sum_ {j = 1} ^ {J} \delta_ {y _ {j}}, P _ {x}\right) \bigotimes_ {j = 1} ^ {J} P _ {x} (\mathrm {d} y _ {j}) \right| \leq \frac {L}{J} \sum_ {j = 1} ^ {J} \| x _ {j} - x \| _ {\infty}.
$$

Proof For  $x, x_1, \ldots, x_J \in \mathbb{X}^{J+1}$ , note that

$$
\begin{array}{l} \left| \int_ {\mathbb {Y} ^ {J}} \mathcal {W} \left(\frac {1}{J} \sum_ {j = 1} ^ {J} \delta_ {y _ {j}}, P _ {x}\right) \bigotimes_ {j = 1} ^ {J} P _ {x _ {j}} (\mathrm {d} y _ {j}) - \int_ {\mathbb {Y} ^ {J}} \mathcal {W} \left(\frac {1}{J} \sum_ {k = 1} ^ {J} \delta_ {y _ {j}}, P _ {x}\right) \bigotimes_ {j = 1} ^ {J} P _ {x} (\mathrm {d} y _ {j}) \right| \\ \leq \sum_ {\ell = 1} ^ {J} \left| \int_ {\mathbb {Y} ^ {J}} \mathcal {W} \left(\frac {1}{J} \sum_ {j = 1} ^ {J} \delta_ {y _ {j}}, P _ {x}\right) \bigotimes_ {j = 1} ^ {\ell - 1} P _ {x} (\mathrm {d} y _ {j}) \otimes \bigotimes_ {j = \ell} ^ {J} P _ {x _ {j}} (\mathrm {d} y _ {j}) \right. \\ \left. - \int_ {\mathbb {Y} ^ {J}} \mathcal {W} \left(\frac {1}{J} \sum_ {j = 1} ^ {J} \delta_ {y _ {j}}, P _ {x}\right) \bigotimes_ {j = 1} ^ {\ell} P _ {x} (\mathrm {d} y _ {j}) \otimes \bigotimes_ {j = \ell + 1} ^ {J} P _ {x _ {j}} (\mathrm {d} y _ {j}) \right|, \\ \end{array}
$$

where for the sake of neatness, at  $\ell = 1,J$  , we set

$$
\bigotimes_ {j = 1} ^ {0} P _ {x} (\mathrm {d} y _ {j}) \otimes \bigotimes_ {j = 1} ^ {J} P _ {x _ {j}} (\mathrm {d} y _ {j}) = \bigotimes_ {j = 1} ^ {J} P _ {x _ {j}} (\mathrm {d} y _ {j}) \quad \text {a n d} \quad \bigotimes_ {j = 1} ^ {J} P _ {x} (\mathrm {d} y _ {j}) \otimes \bigotimes_ {j = J + 1} ^ {J} P _ {x _ {j}} (\mathrm {d} y _ {j}) = \bigotimes_ {j = 1} ^ {J} P _ {x _ {j}} (\mathrm {d} y _ {j}).
$$

Regarding the  $\ell$ -th summand, invoking Fubini-Toneli theorem to integrate  $y_{\ell}$  first then combining the integrals on outer layers using linearity, we obtain

$$
\begin{array}{l} \left| \int_ {\mathbb {Y} ^ {J}} \mathcal {W} \left(\frac {1}{J} \sum_ {j = 1} ^ {J} \delta_ {y _ {j}}, P _ {x}\right) \bigotimes_ {j = 1} ^ {\ell - 1} P _ {x} (\mathrm {d} y _ {j}) \otimes \bigotimes_ {j = \ell} ^ {J} P _ {x _ {j}} (\mathrm {d} y _ {j}) \right. \\ - \int_ {\mathbb {Y} ^ {J}} \mathcal {W} \left(\frac {1}{J} \sum_ {j = 1} ^ {J} \delta_ {y _ {j}}, P _ {x}\right) \bigotimes_ {j = 1} ^ {\ell} P _ {j} (\mathrm {d} y _ {j}) \otimes \bigotimes_ {j = \ell + 1} ^ {J} P _ {x _ {j}} (\mathrm {d} y _ {j}) \\ = \left| \int_ {\mathbb {Y} ^ {J - 1}} \int_ {\mathbb {Y}} \mathcal {W} \left(\frac {1}{J} \sum_ {j = 1} ^ {J} \delta_ {y _ {j}}, P _ {x}\right) \left(P _ {x} - P _ {x _ {\ell}}\right) (\mathrm {d} y _ {\ell}) \bigotimes_ {j = 1} ^ {\ell - 1} \mathrm {d} P _ {x} (y _ {j}) \otimes \bigotimes_ {j = \ell + 1} ^ {J} P _ {x _ {j}} (\mathrm {d} y _ {j}) \right| \\ \leq \sup  _ {(y _ {j}) _ {j \neq \ell} \in \mathbb {Y} ^ {J - 1}} \left| \int_ {\mathbb {Y}} \mathcal {W} \left(\frac {1}{J} \sum_ {j = 1} ^ {J} \delta_ {y _ {j}}, P _ {x}\right) \left(P _ {x _ {\ell}} - P _ {x}\right) (\mathrm {d} y _ {\ell}) \right| \\ \leq \frac {1}{J} \mathcal {W} (P _ {x _ {\ell}}, P _ {x}) \leq \frac {L}{J} \| x _ {\ell} - x \| _ {\infty}, \\ \end{array}
$$

where in the second last inequality we have invoked Kantorovich-Rubinstein duality (cf. (Villani, 2008, Particular case 5.16)) and the fact that, for all  $(y_j)_{j \neq \ell} \in \mathbb{Y}^{J-1}$ , the map  $y_\ell \mapsto \mathcal{W}\left(\frac{1}{J} \sum_{j=1}^J \delta_{y_j}, P_x\right)$  is  $\frac{1}{J}$ -Lipschitz, and where in the last equality, we have used Assumption 2.

We will be using the lemma below, which regards the stochastic dominance between two binomial random variables.

Lemma 17 Let  $n \in \mathbb{N}$  and  $0 \leq p < p' \leq 1$ . Then,  $\operatorname{Binomial}(n,p')$  stochastically dominates  $\operatorname{Binomial}(n,p)$ .

Proof Let  $U_{1},\ldots ,U_{n} \stackrel{\mathrm{i.i.d.}}{\sim}$  Uniform[0,1] and define

$$
H := \sum_ {i = 1} ^ {n} \mathbb {1} _ {[ 0, p ]} (U _ {i}), \quad H ^ {\prime} := \sum_ {i = 1} ^ {n} \mathbb {1} _ {[ 0, p ^ {\prime} ]} (U _ {i}).
$$

Clearly,  $H \sim \operatorname{Binomial}(n,p)$  and  $H' \sim \operatorname{Binomial}(n,p')$ . Moreover, we have  $H \leq H'$ , and thus  $\mathbb{P}(H > r) \leq \mathbb{P}(H' > r)$ , which completes the proof.

# 4.2 Proof of Theorem 7

The proof of Theorem 7 relies the technical lemma below that we state and prove now.

Lemma 18 Let  $p \in [0,1]$  a real number, and let  $M \geq 1$  and  $d \geq 1$  two integers. We then have

$$
\sum_ {m = 1} ^ {M} \binom {M} {m} p ^ {m} (1 - p) ^ {M - m} m ^ {- \frac {1}{d}} \leq ((M + 1) p) ^ {- \frac {1}{d}} + ((M + 1) p) ^ {- 1}.
$$

Proof We compute

$$
\begin{array}{l} \sum_ {m = 1} ^ {M} \binom {M} {m} p ^ {m} (1 - p) ^ {M - m} m ^ {- \frac {1}{d}} = \frac {1}{(M + 1) p} \sum_ {m = 1} ^ {M} \binom {M + 1} {m + 1} p ^ {m + 1} (1 - p) ^ {M - m} (m + 1) m ^ {- \frac {1}{d}} \\ = \frac {1}{(M + 1) p} \sum_ {m = 2} ^ {M + 1} {\binom {M + 1} {m}} p ^ {m} (1 - p) ^ {M + 1 - m} m (m - 1) ^ {- \frac {1}{d}} \\ = \frac {1}{(M + 1) p} \sum_ {m = 2} ^ {M + 1} {\binom {M + 1} {m}} p ^ {m} (1 - p) ^ {M + 1 - m} (m - 1) ^ {1 - \frac {1}{d}} \\ + \frac {1}{(M + 1) p} \sum_ {m = 2} ^ {M} \binom {M + 1} {m} p ^ {m} (1 - p) ^ {M + 1 - m} (m - 1) ^ {- \frac {1}{d}}, \\ \end{array}
$$

where we used that  $m = m - 1 + 1$  in the last equality. Then, using that  $(m - 1)^{1 - \frac{1}{d}} \leq m^{1 - \frac{1}{d}}$  and  $(m - 1)^{-\frac{1}{d}} \leq 1$  for all  $m \geq 2$ , we continue to obtain

$$
\begin{array}{l} \sum_ {m = 1} ^ {M} \binom {M} {m} p ^ {m} (1 - p) ^ {M - m} m ^ {- \frac {1}{d}} \leq \frac {1}{(M + 1) p} \sum_ {m = 2} ^ {M + 1} \binom {M + 1} {m} p ^ {m} (1 - p) ^ {M + 1 - m} m ^ {1 - \frac {1}{d}} \\ + \frac {1}{(M + 1) p} \sum_ {m = 2} ^ {M} \binom {M + 1} {m} p ^ {m} (1 - p) ^ {M + 1 - m} \\ \leq \frac {1}{(M + 1) p} \sum_ {m = 0} ^ {M + 1} \binom {M + 1} {m} p ^ {m} (1 - p) ^ {M + 1 - m} m ^ {1 - \frac {1}{d}} + \frac {1}{(M + 1) p}, \\ \end{array}
$$

where the second term in the last equality are derived from the binomial formula. Finally, introducing a random variable  $V$  with binomial distribution  $\mathcal{B}(M + 1,p)$ , and using Jensen inequality for the concave function  $\mathbb{R}^+ \ni x \mapsto x^{1 - \frac{1}{d}} \in \mathbb{R}^+$ , we obtain

$$
\begin{array}{l} \sum_ {m = 1} ^ {M} \binom {M} {m} p ^ {m} (1 - p) ^ {M - m} m ^ {- \frac {1}{d}} \leq \frac {1}{(M + 1) p} \mathbb {E} \left[ V ^ {1 - \frac {1}{d}} \right] + \frac {1}{(M + 1) p} \\ \leq \frac {((M + 1) p) ^ {1 - \frac {1}{d}}}{(M + 1) p} + \frac {1}{(M + 1) p} = ((M + 1) p) ^ {- \frac {1}{d}} + ((M + 1) p) ^ {- 1}, \\ \end{array}
$$

which conclude the proof.

We are now ready to prove Theorem 7.

Proof [Proof of Theorem 7]

For  $\nu \in \mathcal{P}(\mathbb{X})$ , we obviously have

$$
\mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} (P _ {x}, \hat {P} _ {x} ^ {r}) \nu (\mathrm {d} x) \right] \leq \sup  _ {x \in \mathbb {X}} \mathbb {E} \left[ \mathcal {W} (P _ {x}, \hat {P} _ {x} ^ {r}) \right],
$$

we then focus on proving the right hand side inequality in Theorem 7. To this end, we fix  $x \in \mathbb{X}$  and, to alleviate the notations, we let  $B := \mathcal{B}^r(x)$  as introduced in Definition 5. Let  $N_B := \sum_{m=1}^{M} \mathbb{1}_B(X_m)$ . By Definition 5 and Assumption 3 (i), we have

$$
\begin{array}{l} \mathbb {E} \left[ \mathcal {W} (P _ {x}, \hat {P} _ {x} ^ {r}) \right] = \mathbb {E} \left[ \mathcal {W} (P _ {x}, \hat {\mu} _ {B} ^ {\mathcal {D}}) \right] = \sum_ {m = 0} ^ {M} \mathbb {E} \left[ \mathbb {1} _ {N _ {B} = m} \mathcal {W} (P _ {x}, \hat {\mu} _ {B} ^ {\mathcal {D}}) \right] \\ = \sum_ {m = 1} ^ {M} {\binom {M} {m}} \mathbb {E} \left[ \mathbb {1} _ {X _ {1}, \ldots , X _ {m} \in B} \mathbb {1} _ {X _ {m + 1}, \ldots , X _ {M} \notin B} \mathcal {W} \left(\frac {1}{m} \sum_ {l = 1} ^ {m} \delta_ {Y _ {l}}, P _ {x}\right) \right] + \mathbb {P} \left[ X _ {1}, \ldots , X _ {M} \notin B \right] \mathcal {W} (\lambda_ {\mathbb {Y}}, P _ {x}) \\ \leq \sum_ {m = 1} ^ {M} {\binom {M} {m}} \mathbb {P} \left[ X _ {m + 1}, \ldots , X _ {M} \notin B \right] \mathbb {E} \left[ \mathbb {1} _ {X _ {1}, \ldots , X _ {m} \in B} \mathcal {W} \left(\frac {1}{m} \sum_ {l = 1} ^ {m} \delta_ {Y _ {l}}, P _ {x}\right) \right] + \xi (B ^ {c}) ^ {M} R (0) \\ = \sum_ {m = 1} ^ {M} \binom {M} {m} \mu \left(B ^ {c}\right) ^ {M - m} \int_ {\left(B \times \mathbb {Y}\right) ^ {m}} \mathcal {W} \left(\frac {1}{m} \sum_ {l = 1} ^ {m} \delta_ {y _ {l}}, P _ {x}\right) \bigotimes_ {\ell = 1} ^ {m} \psi \left(\mathrm {d} x _ {\ell} \mathrm {d} y _ {\ell}\right) + \xi \left(B ^ {c}\right) ^ {M} R (0). \tag {22} \\ \end{array}
$$

To compute the integral terms, observe that, for fixed  $m \geq 1$ , by definition of  $R(m)$  in (20), Lemma 16 and Remark 6,

$$
\begin{array}{l} \int_ {(B \times \mathbb {Y}) ^ {m}} \mathcal {W} \left(\frac {1}{m} \sum_ {l = 1} ^ {m} \delta_ {y _ {l}}, P _ {x}\right) \bigotimes_ {\ell = 1} ^ {m} \psi (\mathrm {d} x _ {\ell} \mathrm {d} y _ {\ell}) = \int_ {B ^ {m}} \int_ {\mathbb {Y} ^ {m}} \mathcal {W} \left(\frac {1}{m} \sum_ {l = 1} ^ {m} \delta_ {y _ {l}}, P _ {x}\right) \bigotimes_ {l = 1} ^ {m} P _ {x _ {l}} (\mathrm {d} y _ {l}) \bigotimes_ {\ell = 1} ^ {m} \xi (\mathrm {d} x _ {\ell}) \\ \leq \int_ {B ^ {m}} \left(\int_ {\mathbb {Y} ^ {m}} \mathcal {W} \left(\frac {1}{m} \sum_ {l = 1} ^ {m} \delta_ {y _ {l}}, P _ {x}\right) \bigotimes_ {\ell = 1} ^ {m} P _ {x} (\mathrm {d} y _ {\ell}) + \frac {L}{m} \sum_ {\ell = 1} ^ {m} \| x _ {\ell} - x \| _ {\infty}\right) \bigotimes_ {\ell = 1} ^ {m} \xi (\mathrm {d} x _ {\ell}) \\ \leq \int_ {B ^ {m}} (R (m) + 2 L r) \bigotimes_ {\ell = 1} ^ {m} \xi (\mathrm {d} x _ {\ell}) = (R (m) + 2 L r) \xi (B) ^ {m}. \\ \end{array}
$$

This together with (22) implies that, for any  $x\in \mathbb{X}$

$$
\begin{array}{l} \mathbb {E} \left[ \mathcal {W} (P _ {x}, \hat {P} _ {x} ^ {r}) \right] \leq \sum_ {m = 1} ^ {M} {\binom {M} {m}} \xi (B ^ {c}) ^ {M - m} \xi (B) ^ {m} (R (m) + 2 L r) + \xi (B ^ {c}) ^ {M} R (0) \\ \leq 2 L r + \sum_ {m = 1} ^ {M} \binom {M} {m} \xi \left(B ^ {c}\right) ^ {M - m} \xi (B) ^ {m} R (m) + \xi \left(B ^ {c}\right) ^ {M} R (0). \tag {23} \\ \end{array}
$$

The remainder of the proof is split into three cases. In order to proceed, we will put together (21), Lemma 18, and (23). Below we only keep track of the rate.

- For  $d_{\mathbb{Y}} = 1$ , we have

$$
\begin{array}{l} \mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \hat {P} _ {x} ^ {r}\right) \right] \leq 2 L r + (\xi (B) (M + 1)) ^ {- \frac {1}{2}} + (\xi (B) (M + 1)) ^ {- 1} + (1 - \xi (B)) ^ {M} \\ \leq 2 L r + \left(\underline {{c}} (2 r) ^ {d _ {\mathbb {X}}} (M + 1)\right) ^ {- \frac {1}{2}} + \left(\underline {{c}} (2 r) ^ {d _ {\mathbb {X}}} (M + 1)\right) ^ {- 1} + e ^ {- \underline {{c}} M r ^ {d _ {\mathbb {X}}}}. \\ \end{array}
$$

Controlling the dominating term(s) by setting  $r \sim r^{-\frac{d_{\mathbb{X}}}{2}} M^{-\frac{1}{2}}$ , we yield

$$
r \sim M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}} \quad \mathrm {a n d} \quad \mathbb {E} \left[ \mathcal {W} (P _ {x}, \hat {P} _ {x} ^ {r}) \right] \lesssim M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}}.
$$

- For  $d_{\mathbb{Y}} = 2$ , we have

$$
\begin{array}{l} \mathbb {E} \left[ \mathcal {W} (P _ {x}, \hat {P} _ {x} ^ {r}) \right] \leq 2 L r + \ln (M) (\xi (B) (M + 1)) ^ {- \frac {1}{2}} + (\xi (B) (M + 1)) ^ {- 1} + (1 - \xi (B)) ^ {M} \\ \leq 2 L r + \ln (M) \left(\underline {{c}} (2 r) ^ {d _ {\mathbb {X}}} (M + 1)\right) ^ {- \frac {1}{2}} + \left(\underline {{c}} (2 r) ^ {d _ {\mathbb {X}}} (M + 1)\right) ^ {- 1} + e ^ {- \underline {{c}} M r ^ {d _ {\mathbb {X}}}}. \\ \end{array}
$$

Since  $r \sim \ln(M) r^{-\frac{d_{\mathbb{X}}}{2}} M^{-\frac{1}{2}}$  may not have a closed-form solution, we simply follow the case of  $d_{\mathbb{Y}} = 1$  to yield

$$
r \sim M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}} \quad \mathrm {a n d} \quad \mathbb {E} \left[ \mathcal {W} (P _ {x}, \hat {P} _ {x} ^ {r}) \right] \lesssim M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}} \ln M.
$$

- For  $d_{\mathbb{Y}} \geq 3$ , we have

$$
\begin{array}{l} \mathbb {E} \left[ \mathcal {W} (P _ {x}, \hat {P} _ {x} ^ {r}) \right] \leq 2 L r + (\xi (B) (M + 1)) ^ {- \frac {1}{d _ {\mathbb {Y}}}} + (\xi (B) (M + 1)) ^ {- 1} + (1 - \xi (B)) ^ {M} \\ \leq 2 L r + \left(\underline {{c}} (2 r) ^ {d _ {\mathbb {X}}} (M + 1)\right) ^ {- \frac {1}{d _ {\mathbb {Y}}}} + \left(\underline {{c}} (2 r) ^ {d _ {\mathbb {X}}} (M + 1)\right) ^ {- 1} + e ^ {- \underline {{c}} M r ^ {d _ {\mathbb {X}}}}. \\ \end{array}
$$

By setting  $r\sim r^{-\frac{d_{\mathbb{X}}}{d_{\mathbb{Y}}}}M^{-\frac{1}{d_{\mathbb{Y}}}}$  , we yield

$$
r \sim M ^ {- \frac {1}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}} \quad \text {a n d} \quad \mathbb {E} \left[ \mathcal {W} (P _ {x}, \hat {P} _ {x} ^ {r}) \right] \lesssim M ^ {- \frac {1}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}}.
$$

The proof is complete.

# 4.3 Proof of Theorem 8

Proof [Proof of Theorem 8] We will proceed by using Efron-Stein inequality. Let  $(X_1', Y_1')$  be an independent copy of  $(X_1, Y_1)$ , and define  $\mathcal{D}' := \{(X_1', Y_1'), (X_2, Y_2), \ldots, (X_M, Y_M)\}$ . In view of Assumption 3 (i), by the triangle inequality of  $\mathcal{W}$ , it is sufficient to investigate

$$
\frac {1}{2} M \mathbb {E} \left[ \left(\int_ {\mathbb {X}} \mathcal {W} \left(\hat {\mu} _ {\mathcal {B} ^ {r} (x)}, \hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D} ^ {\prime}}\right) \mathrm {d} \nu (x)\right) ^ {2} \right].
$$

Notice that, by definitions (1),

$$
\left\{\hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D}} \neq \hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D} ^ {\prime}} \right\} \subseteq \left\{X _ {1} \in \mathcal {B} ^ {r} (x) \right\} \cup \left\{X _ {1} ^ {\prime} \in \mathcal {B} ^ {r} (x) \right\}.
$$

Additionally, by definitions (1) again, on the event that  $\left\{\hat{\mu}_{\mathcal{B}^r(x)}^\mathcal{D} \neq \hat{\mu}_{\mathcal{B}^r(x)}^{\mathcal{D}'}\right\}$ , we have

$$
\mathcal {W} \left(\hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D}}, \hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {P} ^ {\prime}}\right) \leq \left(1 + \sum_ {\ell = 2} ^ {M} \mathbb {1} _ {\mathcal {B} ^ {r} (x)} \left(X _ {\ell}\right)\right) ^ {- 1}.
$$

The above together with the condition that  $\nu$  is dominated by  $\lambda_{\mathbb{X}}$  implies that

$$
\begin{array}{l} \mathbb {E} \left[ \left(\int_ {\mathbb {X}} \mathcal {W} \left(\hat {\mu} _ {\mathcal {B} ^ {r} (x)}, \hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D} ^ {\prime}}\right) \nu (\mathrm {d} x)\right) ^ {2} \right] \leq \overline {{C}} ^ {2} \mathbb {E} \left[ \left(\int_ {B (X _ {1}, 2 r) \cup B (X _ {1} ^ {\prime}, 2 r)} \mathcal {W} \left(\hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D}}, \hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D} ^ {\prime}}\right) \lambda_ {\mathbb {X}} (\mathrm {d} x)\right) ^ {2} \right] \\ \leq \overline {{C}} ^ {2} \mathbb {E} \left[ \left(\int_ {B (X _ {1}, 2 r) \cup B (X _ {1} ^ {\prime}, 2 r)} \left(1 + \sum_ {\ell = 2} ^ {M} \mathbb {1} _ {\mathcal {B} ^ {r} (x)} (X _ {\ell})\right) ^ {- 1} \lambda_ {\mathbb {X}} (\mathrm {d} x)\right) ^ {2} \right] \\ \leq 4 \bar {C} ^ {2} \mathbb {E} \left[ \left(\int_ {B (X _ {1}, 2 r)} \left(1 + \sum_ {\ell = 2} ^ {M} \mathbb {1} _ {\mathcal {B} ^ {r} (x)} (X _ {\ell})\right) ^ {- 1} \lambda_ {\mathbb {X}} (\mathrm {d} x)\right) ^ {2} \right] \\ = 4 \overline {{C}} ^ {2} \mathbb {E} \left[ \lambda_ {\mathbb {X}} (B (X _ {1}, 2 r)) ^ {2} \left(\int_ {B (X _ {1}, 2 r)} \left(1 + \sum_ {\ell = 2} ^ {M} \mathbb {1} _ {\mathcal {B} ^ {r} (x)} (X _ {\ell})\right) ^ {- 1} \frac {\lambda_ {\mathbb {X}} (\mathrm {d} x)}{\lambda_ {\mathbb {X}} (B (X _ {1} , 2 r))}\right) ^ {2} \right] \\ \leq 4 \bar {C} ^ {2} (4 r) ^ {2 d _ {\mathbb {X}}} \mathbb {E} \left[ \mathbb {E} \left[ \int_ {B \left(X _ {1}, 2 r\right)} \left(1 + \sum_ {\ell = 2} ^ {M} \mathbb {1} _ {\mathcal {B} ^ {r} (x)} \left(X _ {\ell}\right)\right) ^ {- 2} \frac {\lambda_ {\mathbb {X}} (\mathrm {d} x)}{\lambda_ {\mathbb {X}} \left(B \left(X _ {1} , 2 r\right)\right)} \mid X _ {1} \right] \right], \tag {24} \\ \end{array}
$$

where we have used Jensen's inequality and tower property in the last line. In view of Assumption 3 (i), expanding the inner conditional expectation into an integral with respect to regular conditional distribution (cf. (Bogachev, 2007, Section 10)) then invoking Fubini-Tonelli theorem, we yield

$$
\begin{array}{l} \mathbb {E} \left[ \int_ {B (X _ {1}, 2 r)} \left(1 + \sum_ {\ell = 2} ^ {M} \mathbb {1} _ {\mathcal {B} ^ {r} (x)} (X _ {\ell})\right) ^ {- 2} \frac {\lambda_ {\mathbb {X}} (\mathrm {d} x)}{\lambda_ {\mathbb {X}} (B (X _ {1} , 2 r))} \Bigg | X _ {1} \right] \\ = \int_ {B \left(X _ {1}, 2 r\right)} \int_ {\mathbb {X} ^ {M - 1}} \left(1 + \sum_ {\ell = 2} ^ {M} \mathbb {1} _ {\mathcal {B} ^ {r} (x)} \left(x _ {\ell}\right)\right) ^ {- 2} \bigotimes_ {\ell = 2} ^ {M} \xi (\mathrm {d} x _ {\ell}) \frac {\lambda_ {\mathbb {X}} (\mathrm {d} x)}{\lambda_ {\mathbb {X}} \left(B \left(X _ {1} , 2 r\right)\right)}. \tag {25} \\ \end{array}
$$

For the inner integral in (25), by Assumption 3 (ii), we have

$$
\begin{array}{l} \int_ {\mathbb {X} ^ {M - 1}} \left(1 + \sum_ {\ell = m + 1} ^ {M} \mathbb {1} _ {\mathcal {B} ^ {r} (x)} (x _ {\ell})\right) ^ {- 2} \bigotimes_ {\ell = 2} ^ {M} \xi (\mathrm {d} x _ {\ell}) \\ = \sum_ {\ell = 0} ^ {M - 1} \binom {M - 1} {\ell} \xi \left(\mathcal {B} ^ {r} (x)\right) ^ {\ell} \left(1 - \xi^ {r} \left(\mathcal {B} ^ {r} (x)\right)\right) ^ {M - 1 - \ell} (1 + \ell) ^ {- 2} \\ = \frac{1}{M(M + 1)\xi\big(\mathcal{B}^{r}(x)\big)^{2}}\sum_{\ell = 0}^{M - 1}\binom {M + 1}{\ell + 2}\xi \big(\mathcal{B}^{r}(x)\big)^{\ell +2}\Big(1 - \xi \big(\mathcal{B}^{r}(x)\big)\Big)^{M - 1 - \ell}\frac{\ell + 2}{\ell + 1} \\ \leq \frac{2}{M(M + 1)\xi\big(\mathcal{B}^{r}(x)\big)^{2}}\sum_{\ell = 2}^{M + 1}\binom {M + 1}{\ell}\xi \big(\mathcal{B}^{r}(x)\big)^{\ell}\Big(1 - \xi \big(\mathcal{B}^{r}(x)\big)\Big)^{M + 1 - \ell} \\ \leq \frac {2}{M (M + 1) \xi (\mathcal {B} ^ {r} (x)) ^ {2}}. \\ \end{array}
$$

This together with (24), (25) and Assumption 3 (ii) implies

$$
\mathbb {E} \left[ \left(\int_ {\mathbb {X}} \mathcal {W} \left(\hat {\mu} _ {\mathcal {B} ^ {r} (x)}, \hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D} ^ {\prime}}\right) \mathrm {d} \nu (x)\right) ^ {2} \right] \leq 8 \frac {2 ^ {2 d _ {\mathbb {X}}} \overline {{C}} ^ {2}}{\underline {{c}} ^ {2} M (M + 1)}.
$$

Invoking Efron-Stein inequality, we conclude the proof.

# 4.4 Proof of Theorem 10

In order to prove Theorem 10, we first establish a few technical lemmas. The following lemma is a first step toward finding the average rate of  $k$ -nearest neighbor method.

Lemma 19 Suppose Assumption 2 and 3. Let  $R$  be defined in Section 4.1. Then, for any  $x \in \mathbb{X}$ , we have

$$
\mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \right] \leq R (k) + \frac {L}{k} \sum_ {m = 1} ^ {k} \mathbb {E} \left[ Z _ {x} ^ {(m)} \right],
$$

where  $Z_{(m)}^{x}, m = 1, \ldots, M$  are the order statistics of  $(\|X_m - x\|_{\infty})_{m=1}^{M}$  in ascending order.

Proof We fix  $x \in \mathbb{X}$  for the rest of the proof. By Assumption 3, we have

$$
\begin{array}{l} \mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \right] = M! \mathbb {E} \left[ \mathbb {1} _ {\| X _ {1} - x \| _ {\infty} \leq \| X _ {2} - x \| _ {\infty} \leq \dots \leq \| X _ {M} - x \| _ {\infty}} \mathcal {W} \left(P _ {x}, \frac {1}{k} \sum_ {\ell = 1} ^ {k} \delta_ {Y _ {\ell}}\right) \right] \\ = M! \int_ {(\mathbb {X} \times \mathbb {Y}) ^ {M}} \mathbb {1} _ {\| x _ {1} - x \| _ {\infty} \leq \| x _ {2} - x \| _ {\infty} \leq \dots \leq \| x _ {M} - x \| _ {\infty}} \mathcal {W} \left(P _ {x}, \frac {1}{k} \sum_ {\ell = 1} ^ {k} \delta_ {y _ {\ell}}\right) \bigotimes_ {\ell = 1} ^ {M} \psi (\mathrm {d} x _ {\ell} \mathrm {d} y _ {\ell}) \\ = M! \int_ {\mathbb {X} ^ {M}} \mathbb {1} _ {\| x _ {1} - x \| _ {\infty} \leq \| x _ {2} - x \| _ {\infty} \leq \dots \leq \| x _ {M} - x \| _ {\infty}} \int_ {\mathbb {Y} ^ {k}} \mathcal {W} \left(P _ {x}, \frac {1}{k} \sum_ {\ell = 1} ^ {k} \delta_ {y _ {\ell}}\right) \bigotimes_ {\ell = 1} ^ {k} P _ {x _ {\ell}} (\mathrm {d} y _ {\ell}) \bigotimes_ {j = 1} ^ {M} \xi (\mathrm {d} x _ {\ell}). \\ \end{array}
$$

In view of Lemma 16, replacing  $P_{x_{\ell}}$  above with  $P_{x}$ , we have

$$
\begin{array}{l} \mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \right] \\ \leq M! \int_ {\mathbb {X} ^ {M}} \mathbb {1} _ {\| x _ {1} - x \| _ {\infty} \leq \| x _ {2} - x \| _ {\infty} \leq \dots \leq \| x _ {M} - x \| _ {\infty}} \int_ {\mathbb {Y} ^ {k}} \mathcal {W} \left(P _ {x}, \frac {1}{k} \sum_ {\ell = 1} ^ {k} \delta_ {y _ {\ell}}\right) \bigotimes_ {\ell = 1} ^ {k} P _ {x} (\mathrm {d} y _ {\ell}) \bigotimes_ {j = 1} ^ {M} \xi (\mathrm {d} x _ {\ell}) \\ + \frac {L}{k} \sum_ {\ell = 1} ^ {k} M! \int_ {\mathbb {X} ^ {M}} \mathbb {1} _ {\| x _ {1} - x \| _ {\infty} \leq \| x _ {2} - x \| _ {\infty} \leq \dots \leq \| x _ {M} - x \| _ {\infty}} d _ {\mathbb {X}} (x _ {\ell}, x) \bigotimes_ {j = 1} ^ {M} \xi (\mathrm {d} x _ {\ell}) \\ = \int_ {\mathbb {Y} ^ {k}} \mathcal {W} \left(\frac {1}{k} \sum_ {l = 1} ^ {k} \delta_ {y _ {l}}, P _ {x}\right) \bigotimes_ {\ell = 1} ^ {k} P _ {x} (\mathrm {d} y _ {\ell}) \\ + \frac {L}{k} \sum_ {\ell = 1} ^ {k} M! \int_ {\mathbb {X} ^ {M}} \mathbb {1} _ {\| x _ {1} - x \| _ {\infty} \leq \| x _ {2} - x \| _ {\infty} \leq \dots \leq \| x _ {M} - x \| _ {\infty}} d _ {\mathbb {X}} (x _ {\ell}, x) \bigotimes_ {j = 1} ^ {M} \xi (\mathrm {d} x _ {\ell}). \\ \end{array}
$$

In view of  $R$  defined above (21) and  $Z_{(m)}^{x}$  defined in the statement of this lemma, we conclude the proof.

The next lemma provides an upper bound to  $\sum_{m=1}^{k} \mathbb{E}\left[Z_x^{(m)}\right]$  listed in Lemma 19.

Lemma 20 Let  $Z_{(m)}^{x}$  be defined as in Lemma 19. Under Assumption 3, for any  $x \in \mathbb{X}$ , we have

$$
\sum_ {m = 1} ^ {k} \mathbb {E} \left[ Z _ {(m)} ^ {x} \right] \leq \frac {2}{\underline {{c}} ^ {\frac {1}{d _ {\mathbb {X}}}} d _ {\mathbb {X}}} \frac {M !}{\Gamma (M + \frac {1}{d _ {\mathbb {X}}} + 1)} \sum_ {m = 1} ^ {k} \sum_ {j = 0} ^ {m - 1} \frac {\Gamma (j + \frac {1}{d _ {\mathbb {X}}})}{j !}.
$$

Proof For any  $x \in \mathbb{X}$ , we compute, since  $Z_{(m)}^{x} \in [0,1]$

$$
\mathbb {E} \left[ Z _ {(m)} ^ {x} \right] = \int_ {0} ^ {1} \mathbb {P} \left[ Z _ {(m)} ^ {x} \geq r \right] \mathrm {d} r = \int_ {0} ^ {1} \left(1 - \mathbb {P} \left[ Z _ {(m)} ^ {x} <   r \right]\right) \mathrm {d} r,
$$

and we observe that  $\left\{Z_{(m)}^x < r\right\} = \{N(x,r)\geq m\}$  with  $N(x,r)\coloneqq \sharp \{1\leq m\leq M\mid \| X_m - x\| < r\}$ . We hence have

$$
\mathbb {E} \left[ Z _ {(m)} ^ {x} \right] = \int_ {0} ^ {1} (1 - \mathbb {P} [ N (x, r) \geq m ]) \mathrm {d} r.
$$

Since  $N(x,r)\sim \mathrm{Binomial}(M,\xi (B(x,r)))$  and  $\xi (B(x,r))\geq \underline{c}\lambda_{\mathbb{X}}(B(x,r))\geq \underline{c}\frac{r^{d_{\mathbb{X}}}}{2^{d_{\mathbb{X}}}}$  by Assumption 3 (ii), we obtain that  $\mathbb{P}[N(x,r)\geq m]\geq \mathbb{P}[N'(x,r)\geq m]$  with  $N^{\prime}(x,r)\sim \mathrm{Binomial}(M,\underline{c}\frac{r^{d_{\mathbb{X}}}}{2^{d_{\mathbb{X}}}})$  due to Lemma 17. This implies

$$
\begin{array}{l} \mathbb {E} \left[ Z _ {(m)} ^ {x} \right] \leq \int_ {0} ^ {1} \left(1 - \mathbb {P} \left[ N ^ {\prime} (x, r) \geq m \right]\right) \mathrm {d} r = \int_ {0} ^ {1} \mathbb {P} \left[ N ^ {\prime} (x, r) <   m \right] \mathrm {d} r \\ = \sum_ {j = 0} ^ {m - 1} \binom {M} {j} \int_ {0} ^ {1} \left(\underline {{c}} \frac {r ^ {d _ {\mathbb {X}}}}{2 ^ {d _ {\mathbb {X}}}}\right) ^ {j} \left(1 - \underline {{c}} \frac {r ^ {d _ {\mathbb {X}}}}{2 ^ {d _ {\mathbb {X}}}}\right) ^ {M - j} \mathrm {d} r \\ = \frac {2}{\underline {{c}} ^ {\frac {1}{d _ {\mathbb {X}}}} d _ {\mathbb {X}}} \sum_ {j = 0} ^ {m - 1} \binom {M} {j} \int_ {0} ^ {\frac {c}{2 ^ {\overline {{d}} _ {\mathbb {X}}}}} r ^ {\frac {1}{d _ {\mathbb {X}}} + j - 1} (1 - r) ^ {M - j}   \mathrm {d} r \\ \leq \frac {2}{\underline {{c}} ^ {\frac {1}{d _ {\mathbb {X}}}} d _ {\mathbb {X}}} \sum_ {j = 0} ^ {m - 1} \frac {\Gamma (M + 1)}{\Gamma (j + 1) \Gamma (M - j + 1)} \frac {\Gamma (\frac {1}{d _ {\mathbb {X}}} + j) \Gamma (M - j + 1)}{\Gamma (\frac {1}{d _ {\mathbb {X}}} + M + 1)} \\ = \frac {2 M !}{\underline {{\mathcal {C}}} ^ {\frac {1}{d _ {\mathbb {X}}}} d _ {\mathbb {X}} \Gamma \left(\frac {1}{d _ {\mathbb {X}}} + M + 1\right)} \sum_ {j = 0} ^ {m - 1} \frac {\Gamma \left(\frac {1}{d _ {\mathbb {X}}} + j\right)}{j !}, \tag {26} \\ \end{array}
$$

and the proof is over.

We are now in position to prove Theorem 10.

Proof [Proof of Theorem 10] By combining Lemma 19 and Lemma 20, noting that the upper bound is constant in  $x$ , we have

$$
\sup  _ {x \in \mathbb {X}} \mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \hat {\mu} _ {\mathcal {N} ^ {k} (x)}\right) \right] \leq R (k) + \frac {L}{k} \frac {2 M !}{\underline {{e}} ^ {\frac {1}{d _ {\mathbb {X}}}} d _ {\mathbb {X}} \Gamma \left(M + \frac {1}{d _ {\mathbb {X}}} + 1\right)} \sum_ {m = 1} ^ {k} \sum_ {j = 0} ^ {m - 1} \frac {\Gamma \left(j + \frac {1}{d _ {\mathbb {X}}}\right)}{j !}. \tag {27}
$$

Below we only investigate the rate of the right hand side of (27) as  $M \to \infty$ , and do not keep track of the constant. We first analyze the second term in the right hand side of (27). By Gautschi's inequality (Merkle, 2008, (10.6)), we have

$$
\frac {\Gamma (j + \frac {1}{d _ {\mathbb {X}}})}{j !} = \frac {\Gamma (j + \frac {1}{d _ {\mathbb {X}}})}{\Gamma (j + 1)} \leq j ^ {\frac {1}{d _ {\mathbb {X}}} - 1}, \quad j \in \{0 \} \cup \mathbb {N}.
$$

Thus,

$$
\sum_ {m = 1} ^ {k} \sum_ {j = 0} ^ {m - 1} \frac {\Gamma \left(j + \frac {1}{d _ {\mathbb {X}}}\right)}{j !} \leq \sum_ {m = 1} ^ {k} \sum_ {j = 0} ^ {m - 1} j ^ {\frac {1}{d _ {\mathbb {X}}} - 1} \lesssim \sum_ {m = 1} ^ {k} m ^ {\frac {1}{d _ {\mathbb {X}}}} \lesssim k ^ {1 + \frac {1}{d _ {\mathbb {X}}}}. \tag {28}
$$

By Gautschi's inequality (Merkle, 2008, (12.2)) again, we have

$$
\frac {M !}{\Gamma \left(M + \frac {1}{d _ {\mathbb {X}}} + 1\right)} = \frac {\Gamma (M + 1)}{\Gamma \left(M + \frac {1}{d _ {\mathbb {X}}} + 1\right)} \leq M ^ {- \frac {1}{d _ {\mathbb {X}}}}. \tag {29}
$$

The above implies

$$
\sup _ {x \in \mathbb {X}} \mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \hat {\mu} _ {\mathcal {N} ^ {k} (x)}\right) \right] \lesssim R (k) + M ^ {- \frac {1}{d _ {\mathbb {X}}}} k ^ {\frac {1}{d _ {\mathbb {X}}}}.
$$

We will split the remainder of the proof into three cases.

- For  $d_{\mathbb{Y}} = 1$ , by letting  $k^{-\frac{1}{2}} \sim M^{-\frac{1}{d_{\mathbb{X}}}} k^{\frac{1}{d_{\mathbb{X}}}}$ , we yield

$$
k \sim M ^ {\frac {2}{d _ {\mathbb {X}} + 2}} \quad \text {a n d} \quad \sup  _ {x \in \mathbb {X}} \mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \hat {\mu} _ {\mathcal {N} ^ {k} (x)}\right) \right] \lesssim M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}}
$$

- For  $d_{\mathbb{Y}} = 2$ , since the explicit solution of  $k^{-\frac{1}{2}} \ln k \sim M^{-\frac{1}{d_{\mathbb{X}}} k^{\frac{1}{d_{\mathbb{X}}}}}$  is elusive, we simply follow the configuration derived in the case of  $d_{\mathbb{Y}} = 1$  and yield

$$
k \sim M ^ {\frac {2}{d _ {\mathbb {X}} + 2}} \quad \text {a n d} \quad \sup  _ {x \in \mathbb {X}} \mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \hat {\mu} _ {\mathcal {N} ^ {k} (x)}\right) \right] \lesssim M ^ {- \frac {1}{d _ {\mathbb {X}} + 2}} \ln M.
$$

- For  $d_{\mathbb{Y}} \geq 3$ , by letting  $k^{-\frac{1}{d_{\mathbb{Y}}}} \sim M^{-\frac{1}{d_{\mathbb{X}}}} k^{\frac{1}{d_{\mathbb{X}}}}$ , we yield

$$
k \sim M ^ {\frac {d _ {\mathbb {Y}}}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}} \quad \text {a n d} \quad \sup  _ {x \in \mathbb {X}} \mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \hat {\mu} _ {\mathcal {N} ^ {k} (x)}\right) \right] \lesssim M ^ {- \frac {1}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}}.
$$

The proof is complete.

# 4.5 Proof of Theorem 11

Proof [Proof of Theorem 11] We will proceed by using Efron-Stein inequality. Let  $(X_1', Y_1')$  be an independent copy of  $(X_1, Y_1)$ , and define  $\mathcal{D}' := \{(X_1', Y_1'), (X_2, Y_2), \ldots, (X_M, Y_M)\}$ . In view of Assumption 3 (i), by the triangle inequality of  $\mathcal{W}$ , it is sufficient to investigate

$$
\frac {1}{2} M \mathbb {E} \left[ \left(\int_ {\mathbb {X}} \mathcal {W} \Big (\hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}}} (x)}, \hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}} ^ {\prime}}} ^ {\mathcal {D} ^ {\prime}} (x) \Big) \nu (\mathrm {d} x)\right) ^ {2} \right].
$$

Note that for  $\mathcal{W}\Bigl (\hat{\mu}_{\mathcal{N}^{k,\mathcal{D}_{\mathbb{X}}}(x)}^{\mathcal{D}},\hat{\mu}_{\mathcal{N}^{k,\mathcal{D}_{\mathbb{X}}^{\prime}}(x)}^{\mathcal{D}^{\prime}}(x)\Bigr)$  to be positive, the event  $A_{x}\cup A_{x}^{\prime}$  is necessary, where

$$
A _ {x} := \left\{X _ {1} \in \mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}}} (x) \right\} \quad \text {a n d} \quad A _ {x} ^ {\prime} := \left\{X _ {1} ^ {\prime} \in \mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}}} (x) \right\}.
$$

Moreover,

$$
\mathcal {W} \Big (\hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}} (x)}, \hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}} ^ {\prime}} (x)} ^ {\mathcal {D} ^ {\prime}} (x) \Big) \leq \frac {1}{k}.
$$

It follows that

$$
\begin{array}{l} \mathbb {E} \left[ \left(\int_ {\mathbb {X}} \mathcal {W} \left(\hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}}} (x)}, \hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}} ^ {\prime}}} ^ {\mathcal {D} ^ {\prime}} (x)\right) \nu (\mathrm {d} x)\right) ^ {2} \right] \leq \frac {1}{k ^ {2}} \mathbb {E} \left[ \left(\int_ {\mathbb {X}} \mathbb {1} _ {A _ {x} \cup A _ {x} ^ {\prime}} \nu (\mathrm {d} x)\right) ^ {2} \right] \tag {30} \\ \leq \frac {1}{k ^ {2}} \mathbb {E} \left[ \int_ {\mathbb {X}} \mathbb {1} _ {A _ {x} \cup A _ {x} ^ {\prime}} \nu (\mathrm {d} x) \right] \leq \frac {2}{k ^ {2}} \int_ {\mathbb {X}} \mathbb {P} [ A _ {x} ] \nu (\mathrm {d} x). \\ \end{array}
$$

where the second inequality is due to the fact that the integral value always fall into in  $[0,1]$ , and we have used Fubini-Tonelli theorem and the subadditivity of probability in the third inequality. Regarding  $\mathbb{P}[A_x]$ , by the symmetry stemming from Assumption 3 (i) and the random tie-breaking rule in Definition 9, we have

$$
\mathbb {P} \left[ A _ {x} \right] = \binom {M - 1} {k - 1} \binom {M} {k} ^ {- 1} = \frac {k}{M}.
$$

Consequently,

$$
M \mathbb {E} \left[ \left(\int_ {\mathbb {X}} \mathcal {W} \Big (\hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}}} (x)}, \hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}} ^ {\prime}}} ^ {\mathcal {D} ^ {\prime}} (x) \Big) \nu (\mathrm {d} x)\right) ^ {2} \right] \leq \frac {2}{k}.
$$

Invoking Efron-Stein inequality, we conclude the proof of (5).

We now assume additionally that  $\nu \leq \overline{C}\lambda_{\mathbb{X}}$  to prove the second statement. Following from (30), by using the positivity and subadditivity of indicator functions as well as AM-GM inequality, we have

$$
\begin{array}{l} \mathbb {E} \left[ \left(\int_ {\mathbb {X}} \mathcal {W} \left(\hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}}} (x)}, \hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}} ^ {\prime}}} ^ {\mathcal {D} ^ {\prime}} (x)\right) \nu (\mathrm {d} x)\right) ^ {2} \right] \\ \leq \frac {4}{k ^ {2}} \mathbb {E} \left[ \left(\int_ {\mathbb {X}} \mathbb {1} _ {A _ {x}} \nu (\mathrm {d} x)\right) ^ {2} \right] \leq \frac {4 \overline {{C}} ^ {2}}{k ^ {2}} \mathbb {E} \left[ \left(\int_ {\mathbb {X}} \mathbb {1} _ {A _ {x}} \lambda_ {\mathbb {X}} (\mathrm {d} x)\right) ^ {2} \right] \\ \leq \frac {4 \bar {C} ^ {2}}{k ^ {2}} \int_ {[ 0, 1 ]} \mathbb {P} \left[ \left(\int_ {\mathbb {X}} \mathbb {1} _ {A _ {x}} \lambda_ {\mathbb {X}} (\mathrm {d} x)\right) ^ {2} > \delta \right] \mathrm {d} \delta , \\ \end{array}
$$

where in the second inequality we have used the condition that  $\nu$  is dominated by  $\lambda_{\mathbb{X}}$ , and in the last one the alternative expression of expectation for positive random variables. Let  $\mathbf{Cub}e_{\mathbb{X}}^{\ell}$  be the set of cubes within  $\mathbb{X}$  with edge length  $\iota$ . Since  $\nu$  is dominated by  $\lambda_{\mathbb{X}}$ , with probability 1 we have

$$
A _ {x} = \left\{\text {a t m o s t} (k - 1) \text {o f} X _ {\ell}, \ell = 2, \dots , M, \text {f a l l s i n t o} B _ {x} ^ {\| X _ {1} - x \| _ {\infty}} \right\},
$$

$$
A _ {x} ^ {\prime} = \left\{\text {a t m o s t} (k - 1) \text {o f} X _ {\ell}, \ell = 2, \dots , M, \text {f a l l s i n t o} B _ {x} ^ {\| X _ {1} ^ {\prime} - x \| _ {\infty}} \right\}.
$$

It follows that

$$
\left\{\sum_ {m = 2} ^ {M} \mathbb {1} _ {B} (X _ {m}) > k, \quad \forall B \in \mathrm {C u b e} _ {\mathbb {X}} ^ {\iota} \mathrm {w i t h} \partial B \ni X _ {1} \right\} \subseteq \left\{\int_ {\mathbb {X}} \mathbb {1} _ {A _ {x}} \lambda (\mathrm {d} x) \leq (2 \iota) ^ {d _ {\mathbb {X}}} \right\}.
$$

By combining the above and setting  $\delta = (2\iota)^{2d_{\mathbb{X}}}$ , we yield

$$
\begin{array}{l} \mathbb {E} \left[ \left(\int_ {\mathbb {X}} \mathcal {W} \left(\hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}}} (x)}, \hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}} ^ {\prime}}} ^ {\mathcal {D} ^ {\prime}} (x)\right) \nu (\mathrm {d} x)\right) ^ {2} \right] \\ \leq \frac {4 \bar {C} ^ {2}}{k ^ {2}} \int_ {[ 0, 1 ]} \mathbb {P} \left[ \frac {1}{M - 1} \sum_ {m = 2} ^ {M} \mathbb {1} _ {B} \left(X _ {m}\right) \leq \frac {k}{M - 1}, \exists B \in \mathrm {C u b e} _ {\mathbb {X}} ^ {\frac {1}{2} \delta^ {\frac {1}{d _ {\mathbb {X}}}}} \right] d \delta . \tag {31} \\ \end{array}
$$

In order to proceed, we state and prove a useful technical lemma using the Rademacher complexity technique (cf. (Wainwright, 2019, Section 4)). Below we let  $\mathbf{C u b e}_{\mathbb{X}}$  be the set of cubes inside  $\mathbb{X}$  with edge lengths within [0, 1].

Lemma 21 Let  $X_{2},\ldots ,X_{M}$  be introduced in Assumption 3 (i). For  $\varepsilon \geq 0$

$$
\mathbb {P} \left[ \frac {1}{M - 1} \sum_ {m = 2} ^ {M} \mathbb {1} _ {B} (X _ {m}) \leq \underline {{c}} \lambda_ {\mathbb {X}} (B) - 8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} - \varepsilon , \quad \exists B \in \mathsf {C u b e} _ {\mathbb {X}} \right] \leq \exp \left(- \frac {M - 1}{2} \varepsilon^ {2}\right).
$$

Proof Let  $\pmb{x}^{M} = (x_{2}^{M},\dots,x_{M}^{M})\in \mathbb{X}^{M - 1}$ . To utilize the machinery of Rademacher complexity, we will upper bound the cardinality of the set  $\{\mathbb{1}_B(\pmb{x}^M):B\in \mathsf{Cub e}_{\mathbb{X}}\}$ , where  $\mathbb{1}_B$  applies entrywise. More precisely,  $\mathbb{1}_B(\pmb{x}^M) = (\mathbb{1}_B(x_2^M),\dots,\mathbb{1}_B(x_M^M))$ . To start with, we first note that for  $d = 1,\ldots ,d_{\mathbb{X}}$ , the projected  $(x_{2,d}^{M},\ldots ,x_{M,d}^{M})$  at most separates axis-  $d$  into  $M$  intervals. Additionally, each element in  $\{\mathbb{1}_B(\pmb{x}^M):B\in \mathsf{Cub e}_{\mathbb{X}}\}$  corresponds to selecting two intervals (one for starting and one for ending of the cube) on each axis. Therefore, the cardinality is at most  $M^{2d_{\mathbb{X}}}$ , i.e.,  $\mathsf{Cub e}_{\mathbb{X}}$  has polynomial discrimination  $2d_{\mathbb{X}}$ . It follows from (Wainwright, 2019, Lemma 4.14 and Theorem 4.10) that, for any  $\varepsilon \geq 0$ ,

$$
\mathbb {P} \left[ \sup _ {B \in \operatorname {C u b e} _ {\mathbb {X}}} \left| \frac {1}{M - 1} \sum_ {m = 2} ^ {M} \mathbb {1} _ {B} (X _ {m}) - \xi (B) \right| \geq 8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} + \varepsilon \right] \leq \exp \left(- \frac {M - 1}{2} \varepsilon^ {2}\right).
$$

Finally, in view of Assumption 3 (ii), we conclude the proof of Lemma 21.

In view of (31) and Lemma 21, for  $\delta \in [0,1]$ , we consider  $\varepsilon \geq 0$  such that

$$
\frac {k}{M - 1} = \frac {\underline {{c}} \delta^ {\frac {1}{2}}}{2 ^ {d _ {\mathbb {X}}}} - 8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} - \varepsilon .
$$

Note that this is feasible only if  $\frac{4^{d_{\mathbb{X}}}}{\underline{c}^2}\left(8\sqrt{\frac{2d_{\mathbb{X}}\ln(M)}{M - 1}} +\frac{k}{M - 1}\right)^2\leq 1.$  It follows that

$$
\begin{array}{l} \mathbb {P} \left[ \frac {1}{M - 1} \sum_ {m = 2} ^ {M} \mathbb {1} _ {B} \left(X _ {m}\right) \leq \frac {k}{M - 1}, \forall B \in \operatorname {C u b e} _ {\mathbb {X}} ^ {\frac {1}{2} \delta^ {\frac {1}{2 d _ {\mathbb {X}}}}} \right] \\ \leq \left\{ \begin{array}{l l} 1, & \delta \in \left[ 0, \frac {4 ^ {d _ {\mathbb {X}}}}{\underline {{c}} ^ {2}} \left(8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} + \frac {k}{M - 1}\right) ^ {2} \right], \\ \exp \left(- \frac {M - 1}{2} \left(\frac {c \delta^ {\frac {1}{2}}}{2 ^ {d _ {\mathbb {X}}}} - 8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} - \frac {k}{M - 1}\right) ^ {2}\right), & \delta \in \left(\frac {2 ^ {d _ {\mathbb {X}}}}{\underline {{c}}} \left(8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} + \frac {k}{M - 1}\right) ^ {2}, 1 \right]. \end{array} \right. \\ \end{array}
$$

The above together with (31) implies

$$
\begin{array}{l} \frac {1}{2} M \mathbb {E} \left[ \right.\left(\int_ {\mathbb {X}} \mathcal {W} \Big (\hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}}} (x)}, \hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}} ^ {\prime}}} ^ {\mathcal {D} ^ {\prime}} (x)\right) \nu (\mathrm {d} x)\left. \right) ^ {2} \left. \right] \\ \leq \frac {2 \overline {{C}} ^ {2} M}{k ^ {2}} \left(\frac {4 ^ {d _ {\mathbb {X}}}}{\underline {{c}} ^ {2}} \left(8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} + \frac {k}{M - 1}\right) ^ {2} \right. \\ + \int_ {\frac {2 ^ {d _ {\mathbb {X}}}}{c} \left(8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} + \frac {k}{M - 1}\right)} ^ {1} \exp \left(- \frac {M - 1}{2} \left(\frac {c \eta}{2 ^ {d _ {\mathbb {X}}}} - 8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} - \frac {k}{M - 1}\right) ^ {2}\right) 2 \eta \mathrm {d} \eta\left. \right), \\ \end{array}
$$

where we have performed a change of variable  $\eta = \delta^{\frac{1}{2}}$  in the last line. Relating to exponential and normal density functions, we calculate the integral to obtain

$$
\begin{array}{l} \frac {1}{2} M \mathbb {E} \left[ \left(\int_ {\mathbb {X}} \mathcal {W} \Big (\hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}}} (x)}, \hat {\mu} _ {\mathcal {N} ^ {k, \mathcal {D} _ {\mathbb {X}} ^ {\prime}}} (x) \Big) \nu (\mathrm {d} x)\right) ^ {2} \right] \\ \leq \frac {2 \overline {{C}} ^ {2} M}{k ^ {2}} \frac {4 ^ {d _ {\mathbb {X}}}}{c ^ {2}} \left(\left(8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} + \frac {k}{M - 1}\right) ^ {2} + \frac {\sqrt {2 \pi}}{\sqrt {M - 1}} \left(8 \sqrt {\frac {2 d _ {\mathbb {X}} \ln (M)}{M - 1}} + \frac {k}{M - 1}\right) + \frac {4}{M - 1}\right), \\ \end{array}
$$

where we note the right hand side is of  $O\left(\left(\frac{\sqrt{\ln(M)}}{k} + \frac{1}{\sqrt{M}}\right)^2 + \frac{1}{k}\left(\frac{\sqrt{\ln(M)}}{k} + \frac{1}{\sqrt{M}}\right) + \frac{1}{k^2}\right)$ . Invoking Efron-Stein inequality, we conclude the proof.

# 4.6 Proof of Proposition 13

Proof [Proof of Proposition 13] By triangle inequality,

$$
\begin{array}{l} \mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \tilde {P} _ {x} ^ {\Theta}\right) \mathrm {d} x \right] \\ \leq \mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, P _ {x} ^ {\Theta}\right) \left(\lambda_ {\mathbb {X}} - \frac {1}{N} \sum_ {n = 1} ^ {N} \delta_ {\tilde {X} _ {n}}\right) (\mathrm {d} x) \right] \\ + \mathbb {E} \left[ \frac {1}{N} \sum_ {n = 1} ^ {N} \mathcal {W} (P _ {\tilde {X} _ {n}}, \overline {{P}} _ {\tilde {X} _ {n}}) \right] + \mathbb {E} \left[ \frac {1}{N} \sum_ {n = 1} ^ {N} \mathcal {W} (\overline {{P}} _ {\tilde {X} _ {n}}, \tilde {P} _ {\tilde {X} _ {n}} ^ {\Theta}) \right]. \\ \end{array}
$$

Then, by Assumption 2 and (7),

$$
\mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} (P _ {x}, P _ {x} ^ {\Theta}) \left(\lambda_ {\mathbb {X}} - \frac {1}{N} \sum_ {n = 1} ^ {N} \delta_ {\tilde {X} _ {n}}\right) (\mathrm {d} x) \right] \leq \mathbb {E} \left[ (L + L ^ {\Theta}) \mathcal {W} \left(\lambda_ {\mathbb {X}}, \frac {1}{N} \sum_ {n = 1} ^ {N} \delta_ {\tilde {X} _ {n}}\right) \right].
$$

In view of Assumption 12, we have

$$
\mathbb {E} \left[ \frac {1}{N} \sum_ {n = 1} ^ {N} \mathcal {W} (P _ {\tilde {X} _ {n}}, \overline {{P}} _ {\tilde {X} _ {n}}) \right] = \frac {1}{N} \sum_ {n = 1} ^ {N} \mathbb {E} \left[ \mathbb {E} \left[ \mathcal {W} (P _ {\tilde {X} _ {n}}, \overline {{P}} _ {\tilde {X} _ {n}}) | \tilde {X} _ {n} \right] \right] = \mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} (P _ {x}, \overline {{P}} _ {x}) \mathrm {d} x \right].
$$

Combining the above, we prove the first statement.

As for the second statement, consider  $Q, Q': \mathbb{X} \to \mathcal{P}(\mathbb{Y})$  that are Lipschitz-continuous with constants  $L, L'$ . Suppose that

$$
\mathcal {W} (Q _ {x ^ {*}}, Q _ {x ^ {*}} ^ {\prime}) = \sup  _ {x \in \tilde {\mathbb {X}}} \mathcal {W} (Q _ {x}, Q _ {x} ^ {\prime}) = \delta
$$

for some  $\delta > 0$  and  $x^{*} \in \mathbb{X}$ . This supremum is indeed attainable because  $\mathbb{X}$  is compact that  $x \mapsto \mathcal{W}(Q_x, Q_x')$  is continuous. Consequently, by triangle inequality and the Lipschitz-continuity, we have

$$
\begin{array}{l} \mathcal {W} (Q _ {x}, Q _ {x} ^ {\prime}) \geq \left(\mathcal {W} (Q _ {x}, Q _ {x ^ {*}} ^ {\prime}) - \mathcal {W} (Q _ {x} ^ {\prime}, Q _ {x ^ {*}} ^ {\prime})\right) \vee 0 \\ \geq \left(\mathcal {W} \left(Q _ {x ^ {*}}, Q _ {x ^ {*}} ^ {\prime}\right) - \mathcal {W} \left(Q _ {x ^ {*}}, Q _ {x}\right) - \mathcal {W} \left(Q _ {x ^ {*}} ^ {\prime}, Q _ {x} ^ {\prime}\right)\right) \vee 0 \\ \geq \left(\delta - (L + L ^ {\prime}) \| x - x ^ {*} \| _ {\infty}\right) \vee 0, \quad x \in \mathbb {X}. \\ \end{array}
$$

We may then lower bound  $\int_{\mathbb{X}}\mathcal{W}(Q_x,Q_x')\mathrm{d}x$  with the volume of the cone on right hand side above (note the worst case is when  $x^{*} = (0,0)$ ),

$$
\int_ {\mathbb {X}} \mathcal {W} (Q _ {x}, Q _ {x} ^ {\prime}) \mathrm {d} x \geq \int_ {0} ^ {\delta} \left(\frac {\delta - z}{L + L ^ {\prime}}\right) ^ {d _ {\mathbb {X}}} \mathrm {d} z = \frac {\delta^ {d _ {\mathbb {X}} + 1}}{(d _ {\mathbb {X}} + 1) (L + L ^ {\prime}) ^ {d _ {\mathbb {X}}}}.
$$

It follows that

$$
\sup  _ {x \in \mathbb {X}} \mathcal {W} (Q _ {x}, Q _ {x} ^ {\prime}) \leq \left(d _ {\mathbb {X}} + 1\right) ^ {\frac {1}{d _ {\mathbb {X}} + 1}} \left(L + L ^ {\prime}\right) ^ {\frac {d _ {\mathbb {X}}}{d _ {\mathbb {X}} + 1}} \left(\int_ {\mathbb {X}} \mathcal {W} (Q _ {x}, Q _ {x} ^ {\prime})   \mathrm {d} x\right) ^ {\frac {1}{d _ {\mathbb {X}} + 1}},
$$

which completes the proof.

# 5. Implementation details and ablation analysis

In this section, we will provide further implementation details and conduct ablation analysis of the components highlighted in Section 3.1.

# 5.1 Comparing ANNS-RBSP to exact NNS

Algorithm 3 outlines a single slice of RBSP, which divides an array of  $x$ 's into two arrays of a random ratio along a random axis. Throughout the training, we execute RBSP 5 times during each training epoch, yielding  $2^{5} = 32$  parts. Within each part, we then select a small batch of 8 query points, locating the  $k$  nearest neighbors for each query point within the same part. In Table 1, we compare the execution times of exact NNS and ANNS-RBSP. ANNS-RBSP offers considerable time savings for  $M = 10^{6}$ , while exact NNS is more efficient for  $M = 10^{5}$  or fewer.

Algorithm 3 Single slice of random binary space partitioning  
Input: data  $\mathsf{D}_{\mathbb{X}} = (x_i)_{i=1}^M \subset [0,1]^{d_{\mathbb{X}}}$ , arrays of indexes  $\mathsf{S}_d, d = 1, \ldots, d_{\mathbb{X}}$  of length  $M$  with the  $j$ -th entry indicating the position of the  $j$ -th smallest value in the  $d$ -th dimension of  $\mathsf{D}_{\mathbb{X}}$ , a boolean array  $\mathsf{B}$  of length  $M$  with the  $i$ -th entry indicating whether  $x_i$  is involved in the current slicing, a rectangle  $R$  that bounds  $x_i$ 's involved in the current slicing, i.e.,  $R$  corresponds to  $\mathsf{B}$ , a parameter  $r_{\mathrm{edge}} \in (1,\infty)$  for avoiding thin rectangles, an interval  $[\underline{p}, \overline{p}] \in (0,1)$  for random bisecting ratio  
Output: two boolean arrays  $\mathsf{B}, \mathsf{B}'$  of length  $M$  indicating the bisected data, two bounding rectangles  $R, R'$  that correspond to  $\mathsf{B}, \mathsf{B}'$   
1: Randomly pick a dimension  $d$   
2: if The edge ratio of  $R$  exceeds  $r_{\mathrm{edge}}$  then  
3: Replace  $d$  with that corresponds to the longest edge  
4: end if  
5: Rearrange  $\mathsf{B}$  according to  $\mathsf{S}_d$  by  $\tilde{\mathsf{B}} \gets \mathsf{B}[\mathsf{S}_d]$   
6: Pick out the indexes from  $\mathsf{S}_d$  involved in ANNS by  $\tilde{\mathsf{S}}_d \gets \mathsf{S}_d[\tilde{\mathcal{B}}]$   
7: Generate  $p \sim \mathrm{Uniform}([\underline{p}, \overline{p}])$  and round  $p$  into  $\tilde{p}$  so that  $\tilde{p} \leqslant (\tilde{\mathsf{S}}_d)$  is an integer  
8: Bisect  $\tilde{\mathsf{S}}_d$  in two arrays with length  $\tilde{p} \leqslant (\tilde{\mathsf{S}}_d)$  and  $(1 - \tilde{p}) \leqslant (\tilde{\mathsf{S}}_d)$ , denoted by  $\tilde{\mathsf{S}}_d$  and  $\tilde{\mathsf{S}}_d'$   
9: Form new bounding rectangles  $R, R'$  using  $\tilde{\mathsf{S}}_d, \tilde{\mathsf{S}}_d', \mathsf{D}_{\mathbb{X}}$  and the original  $R$  (may enforce some overlap here)  
10: Initialize two boolean arrays  $\mathsf{B}, \mathsf{B}'$  with length  $M$  and all entries being False  
11:  $\mathsf{B}[\tilde{\mathsf{S}}_d] \gets \mathsf{True}$ ,  $\mathsf{B}'[\tilde{\mathsf{S}}_d'] \gets \mathsf{True}$   
12: return  $\mathsf{B}, \mathsf{B}', R, R'$

It's important to note that ANNS-RBSP may introduce additional errors by inaccurately including points that are not within the  $k$  nearest neighbors. As elucidated in the proof of Theorem 10, the magnitude of this induced error can be understood by comparing the excessive distance

Table 1 Execution times of two NNS methods.  

<table><tr><td colspan="4">k=300</td></tr><tr><td></td><td>M=10^4</td><td>M=10^5</td><td>M=10^6</td></tr><tr><td>d_X=1</td><td>(0.1,9.2)</td><td>(3.7,11.0)</td><td>(19.5,11.2)</td></tr><tr><td>d_X=3</td><td>(0.2,9.8)</td><td>(4.2,11.2)</td><td>(24.6,11.3)</td></tr><tr><td>d_X=10</td><td>(0.4,9.9)</td><td>(5.9,11.4)</td><td>(52.4,11.6)</td></tr><tr><td colspan="4">k=1000</td></tr><tr><td></td><td>M=10^4</td><td>M=10^5</td><td>M=10^6</td></tr><tr><td>d_X=1</td><td>(0.1,7.2)</td><td>(3.8,10.9)</td><td>(19.6,11.1)</td></tr><tr><td>d_X=3</td><td>(0.2,7.2)</td><td>(4.2,11.2)</td><td>(24.7,11.4)</td></tr><tr><td>d_X=10</td><td>(0.4,7.4)</td><td>(5.9,11.4)</td><td>(52.8,11.5)</td></tr></table>

This table compares the execution times for 500 runs of exact NNS versus ANNS-RBSP, both utilizing parallel computing, facilitated by PyTorch, with an NVIDIA L40 GPU. Each iteration (approximately) finds the 300 nearest neighbors from  $M$  samples for all of 256 randomly generated query points. The values within each parenthesis denote the seconds consumed by both methods, with the first number corresponding to exact NNS. For faster processing, exact NNS employs a 3D tensor. ANNS-RBSP regenerates a new partition each run. The table does not include the time required to sort the data along all dimensions, which takes about 0.2 seconds in the worst case and is not repeatedly executed.

incurred to that of exact NNS. For simplicity, we investigate the difference below

$$
\Delta := \frac {1}{N _ {\mathrm {b a t c h}}} \sum_ {i = 1} ^ {N _ {\mathrm {b a t c h}}} \left(\frac {1}{k} \sum_ {j = 1} ^ {k} \left\| \check {X} _ {i j} ^ {\prime} - \tilde {X} _ {i} \right\| _ {1} - \frac {1}{k} \sum_ {j = 1} ^ {k} \left\| \check {X} _ {i j} - \tilde {X} _ {i} \right\| _ {1}\right),
$$

where  $\tilde{X}_i$ 's are query points, and  $\tilde{X}_{ij}, \tilde{X}_{ij}'$  are the  $k$ -nearest-neighbor identified by exact NNS and ANNS-RBSP, respectively. In our experiments, we evaluated scenarios with  $d_{\mathbb{X}} = 3$ , 10 and  $k = 300$ . Regarding the data, we generated  $M = 10^4$ ,  $10^5$ ,  $10^6$  samples from Uniform  $([0,1]^{d_{\mathbb{X}}})$ . Once the data set is generated, we fixed the data and conducted 100 simulations of  $\Delta$ , each with  $N_{\mathrm{batch}} = 256$  query points. This process was repeated 10 times, each with a separately generated data. The results are illustrated in Figure 8. It is expected that  $\Delta$  will approach 0 as the sample size  $M$  tends to infinity. The convergence rate is likely influenced by factors such as  $d_{\mathbb{X}}$ ,  $k$ , and  $N_{\mathrm{batch}}$ . Further analysis of the convergence of ANNS-RBSP will be conducted in future studies.

# 5.2 An implementation of the Sinkhorn algorithm

In this section, we will detail our implementation of the Sinkhorn algorithm and highlight a few novel treatments that seem to enhance the training of the neural estimator. While the mechanisms are not yet fully understood, they constitute important improvement in the accuracy of the neural estimator.

Let us first recall the iterative procedure involved in the Sinkhorn algorithm. We follow the setup in Section 3.1.2. In particular, the row indexes of the cost matrix stand for atoms in the empirical measures, while the column indexes stand for atoms produced by the neural estimator. We set  $N_{\mathrm{atom}} = k$  and let  $\mathsf{u}^{(0)}, \mathsf{v}^{(0)}$  be column vectors of size  $k$  with all entries being  $k^{-1}$ . We will suppress the dependence on  $y$  from the notation. Upon setting

$$
K ^ {\epsilon} := \exp \left(- \frac {C}{\epsilon}\right)
$$

![](images/21b2ef72cf0956ece38043f802071c4555dddc3e79435afeee6d4c1f9633ab29.jpg)

![](images/1f7349e33b8050335aebfb40d8ffb4cdd6c8281aaf7659e6294caf6694ced8fe.jpg)

![](images/be54cee9cabf71ff5ed8f126a20b0a620c3972715816667b7295c7aa7c84b3c5.jpg)  
(a)  $d_{\mathbb{X}} = 3$  
Figure 8: Empirical CDFs of  $\Delta$ .

![](images/3803bb6528e06065df2d912eddc2f31cc582dca1481daff615237a1ffbdcbc59.jpg)  
(b)  $d_{\mathbb{X}} = 10$

We compare the empirical CDFs of  $\Delta$ . Each line corresponding to a independently generated set of data. Each plot includes 10 empirical CDFs. Note the difference in the  $x$  axis scale.

with entry-wise exponential, the Sinkhorn algorithm performs repeatedly

$$
\mathbf {u} ^ {(\ell + 1)} = \frac {\mathbf {u} ^ {(0)}}{\mathsf {K} ^ {\epsilon} \mathbf {v} ^ {(\ell)}} \quad \text {a n d} \quad \mathbf {v} ^ {(\ell + 1)} = \frac {\mathbf {v} ^ {(0)}}{(\mathsf {K} ^ {\epsilon}) ^ {\top} \mathbf {u} ^ {(\ell + 1)}}, \tag {32}
$$

where the division is also calculated entry-wise. After a certain number of iterations, denoted as  $N_{\mathrm{iter}}$ , we obtain an approximate optimal transport plan for problem (15):

$$
\mathsf {T} ^ {\epsilon} = \mathrm {d i a g} (\mathsf {u} ^ {(N _ {\mathrm {i t e r}})}) \mathsf {K} ^ {\epsilon} \mathrm {d i a g} (\mathsf {v} ^ {(N _ {\mathrm {i t e r}})}).
$$

Let us set  $\epsilon = 1$  momentarily. Note that if the entries of  $\mathsf{C}$  are excessively large,  $\mathsf{K}$  effectively becomes a zero matrix, which impedes the computations in (32). This issue may occur at the initiation of the neural estimator or during training, possibly due to the use of stochastic gradient descent. To tackle this issue, we employ a rule-of-thump normalization that

$$
\tilde {\mathsf {K}} ^ {\epsilon} := \exp \left(- \frac {\mathsf {C}}{\tilde {c} \epsilon}\right) \quad \mathrm {w i t h} \quad \tilde {c} := \min _ {i} \max _ {j} \mathsf {C} _ {i j}, \tag {33}
$$

and use  $\tilde{\mathsf{K}}^{\epsilon}$  instead of  $\kappa^{\epsilon}$  in (32). Regarding the selection of  $\epsilon$  and the number of iterations, we currently lack a method for adaptively determining these values. Instead, we adjust them manually based on training episodes. This manual adjustment works well for all models discussed in this paper. For more information, please see Appendix C.

As alluded in Section 3.1.2, we enforce sparsity on the transport plan to improve the performance of the neural estimator. Let  $\tilde{\mathsf{T}}^{\epsilon}$  be the output of the Sinkhorn algorithm. We construct  $\hat{\mathsf{T}}^{\epsilon}$  and  $\check{\mathsf{T}}^{\epsilon}$  by setting the row-wise and column-wise maximum of  $\tilde{\mathsf{T}}^{\epsilon}$  to  $k^{-1}$ , respectively, and setting the remaining entries to 0. We then use

$$
\overline {{\mathsf {T}}} ^ {\epsilon} = \gamma \hat {\mathsf {T}} ^ {\epsilon} + (1 - \gamma) \check {\mathsf {T}} ^ {\epsilon}, \tag {34}
$$

where  $\gamma \in [0,1]$  is a hyper-parameter, in gradient descent (14). We observe that  $\hat{\mathsf{T}}^{\epsilon}$  relates each atom in the empirical measure to a single corresponding atom from the neural estimator, and  $\check{\mathsf{T}}^{\epsilon}$  does the same in reverse. The optimal choice of  $\gamma$  remains an open question, though we have set  $\gamma = 0.5$  in all three models.

Next, we explore the impact of enforcing sparsity and varying the choices of  $\gamma$ . Figure 9 compares the performance in Model 1 under different sparsity parameters. When no sparsity is enforced, the neural estimator tends to handle singularities more adeptly, but may overlooks points located on the periphery of the empirical joint distribution, potentially resulting in overly concentrated atoms from the neural estimator (see around  $x = 0.1, 0.9$ ). Compare Figure 4 and 10 for the extra error due to the lack of enforced sparsity. This phenomenon is more noticeable in Model 3. We refer to panel (2,3) of Figure 18 in Appendix B for an example. Moreover, Figure 19, which is obtained without enforced sparsity, indicates a downgrade in accuracy when compared to Figure 6.

For completeness, we present in Table 2 the accuracy of LipNet across various  $\epsilon$  values, both with and without enforced sparsity. As demonstrated in the table, the improvement resulting from enforced sparsity is evident. It is worth noting that smaller values of  $\epsilon$  generally require more iterations to achieve convergence (see, e.g., (Peyre and Cuturi, 2019, Section 4.2)). Additionally, other hyper-parameters may also significantly influence the training outcomes. Further exploration of these related issues will be conducted in future studies.

Finally, it is not recommended to use  $\overline{\mathsf{T}}^{\epsilon}$  at the early stages of training, as our empirical experiments suggest this could deteriorates performance. In training, we start by not enforcing sparsity and then begin to enforce it in later episodes. We refer to Appendix C for further details of the training configuration.

![](images/f58f5b589fe674d268b19e9cce9ea3eda64b03b77c6232e8b5bccf43cafbc42e.jpg)  
(a) No sparsity enforced

![](images/efd72ac96acf3cb87fbdb68e8c54010b4022c1570d194944a60b2b1b2bcc26b5.jpg)  
(b)  $\gamma = 0$

![](images/8c84d1161242f8930638d41d714471c9afa221651d41b27b2b52bbd3f9782bb9.jpg)  
(c)  $\gamma = 0.5$

![](images/7ed92d8d2f44f50c556e8ed45909a5a4bc475a63e80c06e0f85c0e07dda88ea4.jpg)  
Figure 9: LipNet under Model 1 with different sparsity enforcement.  
$(d)$ $\gamma = 1$

![](images/697e5adb73478004726aab8c5266179a74da2a75109070a3750298aa3b8b6c9b.jpg)  
Figure 10: Errors at different  $x$ 's of various estimators under Model 1, LipNet is trained without enforced sparsity.

We compute the  $\mathcal{W}$ -distance between estimators and the true conditional distribution at different  $x$ 's. The setting is similar to Figure 4, but LipNet is trained without enforcing sparsity on the transport plan. The errors of LipNet at around  $x = 0.1, 0.9$  are slightly higher than those in Figure 4.

Table 2 Projected Wasserstein error under different Sinkhorn parameters.  

<table><tr><td colspan="6">LipNet without enforced sparsity</td></tr><tr><td>ε-1</td><td>mean</td><td>25%-quantile</td><td>median</td><td>75%-quantile</td><td>number of NaN steps</td></tr><tr><td>10</td><td>5.18</td><td>2.18</td><td>3.99</td><td>7.34</td><td>0</td></tr><tr><td>30</td><td>1.43</td><td>0.83</td><td>1.21</td><td>1.78</td><td>0</td></tr><tr><td>100</td><td>1.50</td><td>0.96</td><td>1.31</td><td>1.73</td><td>1217</td></tr><tr><td>300</td><td>9.15</td><td>4.97</td><td>8.62</td><td>12.30</td><td>9500</td></tr><tr><td colspan="6">LipNet with enforced sparsity</td></tr><tr><td>ε-1</td><td>mean</td><td>25%-quantile</td><td>median</td><td>75%-quantile</td><td>number of NaN steps</td></tr><tr><td>10</td><td>1.42</td><td>0.91</td><td>1.20</td><td>1.60</td><td>0</td></tr><tr><td>30</td><td>1.24</td><td>0.79</td><td>1.07</td><td>1.42</td><td>0</td></tr><tr><td>100</td><td>1.52</td><td>0.97</td><td>1.32</td><td>1.76</td><td>0</td></tr><tr><td>300</td><td>5.00</td><td>4.96</td><td>5.01</td><td>5.05</td><td>0</td></tr></table>

<table><tr><td colspan="5">Raw estimator</td></tr><tr><td>k</td><td>mean</td><td>25%-quantile</td><td>median</td><td>75%-quantile</td></tr><tr><td>300</td><td>2.69</td><td>1.62</td><td>2.22</td><td>3.23</td></tr><tr><td>1000</td><td>1.57</td><td>0.95</td><td>1.30</td><td>1.91</td></tr><tr><td>3000</td><td>1.41</td><td>0.88</td><td>1.21</td><td>1.70</td></tr><tr><td>10000</td><td>2.38</td><td>1.45</td><td>2.02</td><td>2.97</td></tr></table>

This table displays the evaluation error of LipNet across various values of  $\epsilon$ , both with and without enforced sparsity. For comparison, the error of the raw  $k$ -nearest-neighbor estimator is also included. The mean and quantiles are reported at a scale of  $10^{-2}$ . Boldfaced numbers indicate the best results, while underlined numbers denote the runner-up. Note that  $\epsilon$  is only applied after the first 500 training episodes. For additional details on the configuration, please refer to Table 4. The evaluation is performed by computing the projected Wasserstein distance, as outlined in Section 3.2.

# 5.3 More on LipNet

We will investigate the impact of various hyper-parameters on the performance of LipNet. The LipNets presented in this section are trained with the same hyper-parameters as in Section 3.2 (see also Appendix C), expect for those specified otherwise.

# 5.3.1 ACTIVATION FUNCTION

Switching the activation function from ELU to Rectified Linear Unit (ReLU) appears to retain the adaptive continuity property. In Figure 11, we illustrate the joint distribution and the average absolute derivatives of all atoms of LipNet with ReLU activation. The outcomes are on par with those achieved using ELU activation as shown in Figure 2.

# 5.3.2 VALUE OF  $L$  IN (18)

Note that the LipNets discussed in Section 3.2 were trained with  $L = 0.1$ . If the normalizing constants in LipNet are exactly computed,  $L$  reflects the Lipschitz constant of LipNet, up to the discrepancy in the choice of norms in different layers. The effect of  $L$  in our implementation, however, is rather obscure. Figure 12 showcases the performance of LipNets across various  $L$  values in Model 1. The comparison in Model 2 is presented in Figure 20 in Appendix B. The best choice of  $L$  appears to depend on the ground truth model. For Model 3, we compared the performance of  $L = 0.1$  and  $L = 1$  and observed no significant differences. Generally, we prefer a smaller  $L$ ;

![](images/193f2b4302bc6b55858aef5706a35a9a4b52c9d6a15a1ad10b4291e7867370dd.jpg)  
(a) Model 1, all atoms

![](images/c7edb3962826f74ea9a93c068d805e388e4f0b48e72aaa9d2ac67971679f8204.jpg)  
(b) Model 2, ave. abs. der.

![](images/68f7eef81994f5f3e66ccea7c6214022b8f4a52e7c86ebeca208a92c022f5094.jpg)  
Figure 11: LipNet under Model 1 with ReLU activation.

![](images/d90bb2ea4c1d75841dd07ebeb47b58085f6de01f1123db60a83202db8a3b4c5c.jpg)  
(c) Model 2, all atoms  
$(d)$  Model 2, ave. abs. der.

however, smaller values of  $L$  tend to exhibit greater sensitivity to other training parameters. For instance, in Model 3, with  $L = 0.1$ , starting enforcing sparsity too soon leads to significantly poorer performance, while the impact on the outcomes for  $L = 1$  is much less noticeable.

![](images/143ab6ba8eac1960fe13dfa1a9750de31f285537e13bf4ca91f59992d9c79539.jpg)

![](images/6724618dc2651e9911c665a2a90ba43638e54507e36648148b1f3a059bab2bf8.jpg)  
(a)  $L = 0.01$  
Figure 12: LipNet under Model 1 with various  $L$ 's.

![](images/d8649f08c526c0f29330bf6fb8695451b7e96323061b7a579a23334d273e5601.jpg)

![](images/238712f23af99b8bcd6145a2c2f9c405bf73444f43d2730cf0c3875bc1088268.jpg)  
(b)  $L = 0.03$

![](images/b8d20149cef66a5c87ee27a3d2d14118845820aa8c16631400b4e1d0aed324b0.jpg)

![](images/f865a56b81cd3e9f7b8fd49a5526720bb940601c1b0a8bb83c5b164770c299a1.jpg)  
(c)  $L = 1$

![](images/8fd3554578b0648a5da36514095c0abe7614baeffbb14d8e844d2fce053e67ad.jpg)

![](images/93692e6b0dcf45f0b6433c7338bfad5e28b52e864b17cde881df740ff16c5ba7.jpg)  
(d)  $L = 3$

# 5.3.3 MOMENTUM  $\tau$  IN ALGORITHM 2

In our training of LipNet, we use  $\tau = 10^{-3}$ . Figure 13 demonstrates the impact of various  $\tau$  values on neural estimator's performance in Model 1. It is clear that the performance declines with a  $\tau$  that is too large. While we initially speculated that a smaller  $\tau$  might cause atoms to exhibit more erratic movements as  $x$  changes, observations contradict this hypothesis. We now believe that a suitable  $\tau$  value helps prevent neurons from stagnating in the plateau region of the ELU activation function. This is supported by the outcomes observed with  $\tau = 10^{-6}$ , where atom movements are overly simplistic. Additional comparisons in Model 2 are presented in Figure 21.

Despite considering as a potential improvement the inclusion of batch normalization in the convex potential layer (16), right after the affine transformation, along with a corresponding offset in the position of  $\| W\| _2$ , our experiments with both ELU and ReLU activations, using the default

![](images/68b1ef133bab65c829c72c9694285bbe98809e3568dd587cef6b3ef582e1eac6.jpg)

![](images/741385d70f9a4e383f5fcfacc801700b7ab05e1ff02845086d0d03e8849066cb.jpg)

![](images/4ee4490880073be2371ee7d762a7a2eca81935a0b1497172cba90b07d96800ee.jpg)

![](images/9313e6e58acb3d964aceec83781f59da005cc80976b9ea04dc3fb15b1e3486e2.jpg)

![](images/f590eec22de8786c470638761eee701e1c101be10dab354a2e12ca38834f354b.jpg)  
(a)  $\tau = 10^{-1}$

![](images/658a19494aad373fe3389f88969ed301f0f37c0597c858f3eef1be50b1740e29.jpg)  
(b)  $\tau = 10^{-2}$

![](images/ca644cfe0b978c867df6ee4e7acb6ac4868c77a5bb7a3febf307a08a94d7e3ec.jpg)  
(c)  $\tau = 10^{-5}$

![](images/2ff0d07bf716ff4ef2280a193fd0ffc119da73bcb6ea54ef4a20095dd7e0bd31.jpg)  
Figure 13: LipNet under Model 1 with various  $\tau$ 's.  
$(d)$ $\tau = 10^{-6}$

batch normalization momentum of 0.1, resulted in reduced performance. Lowering said batch normalization momentum often leads to a NaN network.

# 6. Weakness and potential improvement

In this section, we provide some discussion on the weakness and possible improvement of our implementation in Section 3.

Extra correction. In more realistic scenarios, the true conditional distribution is often unknown or intractable. In such cases, it is unclear whether a neural estimator offers extra correction over raw estimators. A potential solution to this issue is to train StdNet and LipNet simultaneously. If StdNet and LipNet align more closely with each other than with the raw estimator involved in their training, it is possible that the neural estimators are providing extra corrections.

Hyper-parameters for Sinkhorn algorithm. Our implementation of the Sinkhorn algorithm involves several hyper-parameters: (i)  $k$  in Definition 9; (ii)  $N_{\mathrm{atom}}$  in (10); (iii)  $\epsilon$  in (33); (iv)  $\gamma$  in (34); and (v) additional hyper-parameters listed in Section C. The impact of these hyper-parameters is not yet fully understood. Additionally, an adaptive  $\epsilon$  that balances the accuracy and stability of the Sinkhorn iteration is desirable. Furthermore, as illustrated in Section 5.2, enforcing sparsity on the transport plan generally yields better approximations at  $x$  where the conditional distribution is more diffusive, but may performs worse where the conditional distribution exhibits atoms. This observation motivates further investigation into a sparsity policy that adjusts according to the indications from the raw estimator.

Adaptive continuity. The impact of hyper-parameters in LipNet also warrants further investigation. In addition, despite the results presented in this study, more evidence is needed to understand how LipNet and its variations perform under various conditions.

Scalability. While the implementation produces satisfactory results when  $M$  and  $k$  are relatively small (recall that we set  $N_{\mathrm{atom}} = k$ ), our further experiments indicate a scalability bottleneck. For example, in Model 1, significantly increasing  $M$  and  $k$  does not necessarily improve the performance of neural estimators in a comparable manner. For completeness, we provide a report of the computational time consumption under various settings in Section C. To address this issue, we

could experiment with varying the ratios between  $N_{\mathrm{atoms}}$  and  $k$ , rather than setting them equal, in hopes of reducing the strain on the Sinkhorn algorithm. We note that varying the ratio between  $N_{\mathrm{atoms}}$  and  $k$  requires adjusting the enforced sparsity accordingly. Another issue relates to the dimensions of  $\mathbb{X}$  and  $\mathbb{Y}$ . In view of the curse of dimensionality in Theorem 10, our method is inherently suited for low-dimensional settings. Fortunately, in many practical scenarios, the data exhibits low-dimensional structures, such as: (i) the sampling distribution of  $X$  concentrating on a low-dimensional manifold; and (ii) the mapping  $x \mapsto P_x$  exhibiting low-dimensional dependence. For (i), we might resort to dimension reduction techniques, although an extension of the results in Section 2 has yet to be established. For (ii), a data-driven method that effectively leverages the low-dimensional dependence is of significant interest.

Conditional generative models. Utilizing a conditional generative model could potentially lead to further improvements. One advantage of conditional generative models is the ease of incorporating various training objectives. For instance, it can easily adapt to the training objectives in (6) to accommodate multiple different hyper-parameters simultaneously. We may also incorporate the joint empirical measure in the training process. This flexibility also allows for the integration of specific tail conditions as needed.

Lastly, we would like to point out an issue observed in our preliminary experiments when utilizing a naive conditional generative model: it may assign excessive probability mass to the blank region between two distinct clusters (for example, in Model 1 around  $(x,y) = (0.1,0.5)$ ). This possibly stems from the inherent continuity of neural networks. One possible solution is to consider using a mixture of multiple conditional generative models.

# Appendix A. Numerical finite-sample bounds

In this section, we present numerical computations of finite-sample bounds for the  $r$ -box and  $k$ -nearest-neighbor estimators in the case where  $d_{\mathbb{X}} = d_{\mathbb{Y}} = 3$  and  $\overline{C} = \underline{c} = 1$ , i.e., both the sampling distribution of  $X$  and  $\nu$  are uniform. Through this analysis, we also aim to explore the impact of the Lipschitz constant  $L$  and the hyperparameters  $r$  and  $k$ , as introduced in Sections 2.2 and 2.3.

To begin, we note that an explicit expression for  $R(m)$ , as defined in (21), can be found in (Kloeckner, 2020, Theorem 2.1). Additionally, we introduce a parameter  $\sigma$  to represent the level of uncertainty in  $P_{x}$ . This uncertainty parameter is motivated by the observation that  $R(m)$  is derived as a bound across all probability measures on  $[0,1]^{d_{\mathbb{Y}}}$ . In scenarios where, for each  $x \in \mathbb{X}$ ,  $P_{x}$  is supported on an cube (possibly dependent on  $x$ ) of diameter  $\sigma^{\frac{1}{d_{\mathbb{Y}}}} \in (0,1)$ , we can replace  $R(m)$  in (23) and (27) with  $\sigma R(m)$ . A similar approach applies to other cases, such as small moment conditions. The precise definition of the uncertainty parameter  $\sigma$  and its data-driven estimation will be explored in future work. Here, we use  $\sigma$  to investigate how the magnitude of uncertainty affects the bounds and the corresponding optimal configurations.

For the  $r$ -box estimator, we utilize (23). In the current setting,  $\xi(B)$  in (23) simplifies to  $(2r)^{d_{\mathbb{X}}}$ . Consequently, the bound becomes

$$
\mathbb {E} \left[ \mathcal {W} (P _ {x}, \hat {P} _ {x} ^ {r}) \right] \leq 2 L r + \sigma \sum_ {m = 1} ^ {M} {\binom {M} {m}} \left(1 - (2 r) ^ {d _ {\mathbb {X}}}\right) ^ {M - m} (2 r) ^ {d _ {\mathbb {X}} m} R (m) + \sigma \left(1 - (2 r) ^ {d _ {\mathbb {X}}}\right) ^ {M} R (0).
$$

The second term on the right-hand side can be approximated using Monte Carlo sampling from a binomial distribution with  $n = M$  and  $p = (2r)^{d_{\mathbb{X}}}$ . The Monte Carlo approximation is performed with 1,000 samples.

For the  $k$ -nearest-neighbor estimator, we combine (27) with (28) and (29) to obtain the following upper bound

$$
\mathbb {E} \left[ \mathcal {W} \left(P _ {x}, \hat {\mu} _ {\mathcal {N} ^ {k} (x)}\right) \right] \leq \sigma R (k) + \frac {2 L}{k} M ^ {- \frac {1}{d _ {\mathbb {X}}}} \sum_ {m = 1} ^ {k} \sum_ {j = 0} ^ {m - 1} j ^ {\frac {1}{d _ {\mathbb {X}}} - 1}.
$$

Table 3 below reports the optimal values and configurations of  $r$  and  $k$  based on these bounds, with different  $L$  and  $\sigma$ . For visualization, these bounds are also plotted in Figure 14.

Table 3 Finite sample bounds with  $d_{\mathbb{X}} = d_{\mathbb{Y}} = 3$ $\overline{C} = \underline{c} = 1$ $M = 10^{6}$  

<table><tr><td colspan="5">σ = 0.1</td></tr><tr><td></td><td colspan="2">r-box</td><td colspan="2">k-NN</td></tr><tr><td>L</td><td>min. bound (10-2)</td><td>r*</td><td>min. bound (10-2)</td><td>k*</td></tr><tr><td>0.1</td><td>3.79</td><td>0.095</td><td>4.49</td><td>3548</td></tr><tr><td>0.3</td><td>6.57</td><td>0.055</td><td>7.57</td><td>707</td></tr><tr><td>1</td><td>11.00</td><td>0.005</td><td>11.90</td><td>10</td></tr></table>

<table><tr><td colspan="5">σ = 1</td></tr><tr><td></td><td colspan="2">r-box</td><td colspan="2">k-NN</td></tr><tr><td>L</td><td>min. bound (10-2)</td><td>r*</td><td>min. bound (10-2)</td><td>k*</td></tr><tr><td>0.1</td><td>12.00</td><td>0.3</td><td>14.53</td><td>112201</td></tr><tr><td>0.3</td><td>20.78</td><td>0.175</td><td>24.96</td><td>22387</td></tr><tr><td>1</td><td>37.94</td><td>0.095</td><td>44.85</td><td>3548</td></tr></table>

![](images/4cce3aea4d278d3f9e5160c97efc5c592d3a59dac97bf73432442a510428fcc8.jpg)

![](images/9c51561c0a5bb541cfd38489f8d6401d8f8c7fde11d21b82933011171be646a3.jpg)

![](images/3878dc8ed5fe68be30b00fd4f49148234168e96d77fa49ca3621832bb444e63d.jpg)  
Figure 14: Finite sample bounds with  $d_{\mathbb{X}} = d_{\mathbb{Y}} = 3$ ,  $\overline{C} = \underline{c} = 1$ ,  $M = 10^{6}$ .

![](images/58ba28b5eb80058dfe5c902689ae61140cb5d3a6b4fe241db2fcca352eff38c8.jpg)

![](images/7973e44a91dcabc85491560d15b2db128248e80977533af17fc85d9008e58dab.jpg)

Note that the error is capped at 1, as the finite sample bounds are calculated for unit boxes. For the minimal values, we refer to Table 3.

# Appendix B. Additional plots

In this section, we present additional plots to further support and complement the discussions provided earlier.

![](images/5c71512b3eb57eb83153802d82c37b0b845bea09d031eb30ef7cc03c263f3b83.jpg)

![](images/528c1817264bcc3b1e0376f117d2aa8815f18bbcf1e80106293101f0bd813d67.jpg)

![](images/f1e09d6a4beaece6095999b68444cdf8f68159e92364e086f775e52c6d826378.jpg)

![](images/e9b0f5acef379c435251e9485991b6860ca99a03858aa44d25c11b16440df51f.jpg)

![](images/889b1280b1d43c0ee4b84abef9cefd128f0f24663bb99aafcdd5a2e84e18f034.jpg)

![](images/69153a94be71507264ccf3f81b01bc7207667a3b01dc67e44fde939f7578c7cc.jpg)

![](images/26c7355fa621a8fce2b36439db038359915df5ce2d0f11f17e442146abc63588.jpg)

![](images/3311648b6245f6d4205c7634920711f364b0b62670470bb5917ab8c7b79dca80.jpg)

![](images/a53c2d6b9c3fe400d874f6ef043d7ffe40878e3f3f93afeabb9455e52459be65.jpg)  
Figure 15: Various estimators under Model 3, projections of conditional CDFs,  $k = 1000$  for  $k$ -NN estimator.

![](images/b65fd1cb6e7eb146352ffa294e0f19b33fe54b08e27a4360e50d8f4bc86bfa67.jpg)

![](images/2bc430974e3f0bea17602a53601e514eee6fee4db6f9159ea80488f31af612c6.jpg)

![](images/2734496cb57f789bcd12c0a9916116d90c4fba9d3cab4b7c80b05597527d8e15.jpg)

![](images/8d205718d1c850d9edb1a279eb24a5e53377e3b9ddcfcd7b398750f68cd7c507.jpg)

This follows the setting of Figure 5, expect that we set  $k = 1000$  for the  $k$ -NN estimator plotted here. The neural estimator is trained under the same setting as that in Figure 5 Plot titles display the vectors used for projection. Note the difference in the  $x$  axis scale.

# Appendix C. Additional Implementation Details

Table 4 below summarizes hyper-parameters of the neural networks. Additionally, we report the time consumption in Table 5 under various settings. All timings were obtained from a machine equipped with an Nvidia L40 GPU. It is important to note that the computational time does not fully reflect the curse of dimensionality, as the flexibility in choosing the hyper-parameter  $k$  allows for adjustments in practice. The appropriate selection of  $k$ , as discussed in Section A, is a nuanced issue and needs to be carefully re-evaluated based on the specific field information at hand.

![](images/15285635802c67015d500378622b960df541ff2c8be5bd4bc9f4e54a1a710b23.jpg)

![](images/3a334879511350784e5aebc59d5e005716f366a83a69929073ad9f7f8625dfc3.jpg)

![](images/75c42c41e91b845ed5249b9a86e883836f4e221f931e631d6d18cfbe7d7522e5.jpg)

![](images/194aa8bfb06f13ce175318db54571d0639bdb7fd3e025bdc6c95af3d078857a8.jpg)

![](images/ce679367aede11bee378463eaf19d71a12275a85746cbdf51bdfd2b151e06ba3.jpg)

![](images/e0cefc0bb6722f3e330706c732f75a9e2a1ab9b6e6dc83875deb2934d7b3ee5d.jpg)

![](images/bceca8a015ed757d60a9c4c759969d7d64858aa5d7dc60d99ca3487ef603422f.jpg)

![](images/f55291a81c84ddac5fa84ab80916ca063a02ccd7fde60a78e8c186f399da2dd1.jpg)

Figure 16: Various estimators under Model 3, projections of conditional CDFs,  $k = 3000$  for  $k$ -NN estimator.  
![](images/706047b99e47d1de5be68b3181736e63f1231ac1e529da1222146a9fb83da9b5.jpg)  
—— StdNet —— LipNet —— Raw, k=3000 —— True

![](images/df0cc6e87eec6c5c620df7f46cf9ece3ee1e2208e726e7c3b82aad0ec0d7526a.jpg)

![](images/ea4406c997ee3cb3755750d79e67833233a059b6cdd8bec688323a0728fe7d94.jpg)

![](images/0bf69f03e7fa711131f32c69a6198c5d09221e81b7269f8d871c30a2fc0dbd79.jpg)

This follows the setting of Figure 5, expect that we set  $k = 3000$  for the  $k$ -NN estimator plotted here. The neural estimator is trained under the same setting as that in Figure 5 Plot titles display the vectors used for projection. Note the difference in the  $x$  axis scale.

![](images/b7be6ad340e64792b43f3912b21cb333c356e2cf5183ddce9c5f1b9b31974374.jpg)

![](images/e4f4f6c44f13830039958b3debfdeb86dced26685ac8c3a6285d842e8e20be97.jpg)

![](images/dd5429c09456c63c62ae5c3fa55970e86231d952545d7c0a140119dd4dd43a5d.jpg)

![](images/318ae15607b4882b5df20723e14a93d32cc59696b43f0bf76236ba47d1bf2bc7.jpg)

![](images/1683bab2d18d1c27283b9a6477c856262cbed1b5a5a8bf0d6c0290b0b0d57fa0.jpg)

![](images/587b8c9c4ccd26387836f5cade2258ef222ae8e447edcd9a87bcf9da8b7a469c.jpg)

![](images/8c6c55e717b8a4d42805280f6c652b13bb78f909d283e68bad424e9c17e2fc8f.jpg)

![](images/115b74b54e46607c7ba4e47d2a5c354ed785329fb6279684262fdde4c122297b.jpg)

Figure 17: Various estimators under Model 3, projections of conditional CDFs,  $k = 10^4$  for  $k$ -NN estimator.  
![](images/5909d7a4e3d719590dd2e76938a246620c280cb130230b0749d75a45d0b657da.jpg)  
—— StdNet —— LipNet —— Raw, k=10000

![](images/70414d57cf5bab11865188544ae8050e1b6714e2f9a1fece14fdfa8c9816b25f.jpg)

![](images/4d51173792dd47bfb4cfcb488744f5d9b953737df68474ff0d75eb6cd31a88d9.jpg)

![](images/13add2a5c9f137951eded59ae3a6f88bb0c4e19918675447ab1c6be9a77ab91d.jpg)

This follows the setting of Figure 5, expect that we set  $k = 10^4$  for the  $k$ -NN estimator plotted here. The neural estimator is trained under the same setting as that in Figure 5 Plot titles display the vectors used for projection. Note the difference in the  $x$  axis scale.

![](images/87583183e4ab9f34e2dd399466ba7b56da5f35e7912d1d0c21b98a59fab649de.jpg)

![](images/c39ff1f7772803349f73cc2d1ce96e54a9d7f6e6587840bad2c27497f49e1168.jpg)

![](images/eebf0b7fe29bf7c637dddb1e0813bd91d2ed44d6d70679d1fe21ad520b8630dc.jpg)

![](images/7bfb2cbd51343b1d436621c62eebe6de025d4916a5efdd82a1ef4bd4cbed8021.jpg)

![](images/68101de61f07779d45dcbba8bb930231eecfb426e4f3c6bc8abc0c2b35b1c754.jpg)

![](images/03d69895bde8ec96ea16c51ab1e0d3f0bab9551236da05b30eb28f7fcade705f.jpg)

![](images/81cae1db8c458c1e61e38c9ee3151e4d704e4b0887033feefcd282418aa2c898.jpg)

![](images/590f353b6a4d0636ad66026ceaadd941647551c1455682b3e8bc662486a1a4d5.jpg)

![](images/8fa357868843d04254b4775eaf06642b0cda484f1db26c749192adc8cc524383.jpg)  
Figure 18: Various estimators under Model 3, projections of conditional CDFs, both StdNet and Lipnet are trained without enforced sparsity.

![](images/86b36e4d43d2d459899cd91d134adeff644ade14b64217f30bcddce069fce35c.jpg)

![](images/8026b2068fb0c642c69ec4bcd806722fba71b2a093005bed9f56fa54d129a9c4.jpg)

![](images/2188f1ae79af0b3d2674974f5e76cf1b195cc7491afa9152f42bc7fd29498351.jpg)

![](images/c176ef2e74568811bc1d2d23cda5dd6661edf8a2828d318df5cc8bdee396b274.jpg)

This follows the setting of Figure 5, expect that we do not enforce sparsity on the transport plan during the training of the neural estimator. Plot titles display the vectors used for projection. Note the difference in the  $x$  axis scale.

![](images/6b3f1381993f8cf4f58c9d7761be975679b7829956aa4891326101bd3003be31.jpg)  
Raw  $k = 1000$

![](images/10d5f5115449a7637d1b2bda91cf5aaa508765a687027fcdddd57266d71b2e61.jpg)  
Raw  $k = 3,000$

![](images/0161ed8dbddb2de52ccc10bd3d12ee7d182e7cba01aa4dfe0b985e0cd430ed5e.jpg)  
StdNet

![](images/b054ca2875f4e3fe809dbe323547a9a0e0f445f7fef30f0010b04add63cea09c.jpg)  
Figure 19: Histogram of 10,000 projected Wasserstein-1 errors, without enforced sparsity on transport plan.  
LipNet

This follows the setting of Figure 6, expect that we do not enforce sparsity on the transport plan during the training of the neural estimator. Histograms for raw estimators remain the same. The histogram consists of 20 uniformly positioned bins between 0 to 0.1. The errors of different estimators are computed with the same set of query points and projection vectors. Errors larger than 0.1 will be placed in the right-most bins.

![](images/de82eae6677e9aed2a2edd9ba683f54fc37724fb2d075057b98e4ff84250f0b2.jpg)

![](images/65537943aa5022d3899f7949738d22f752dfe1b29826506f04d8ba78021de4f4.jpg)

![](images/e292d16231f63e06496e73ec9cdc4a30b8b2a009250da8771cd368418f53c677.jpg)

![](images/a4bcfc0ada633b730fd74efb37b12a8690bce102b4223858880940fae6b0cd40.jpg)

![](images/2eb47837b189d72ad72360dca50515d769c7330b53b253e9bbe4da54a85420d1.jpg)  
(a)  $L = 0.01$

![](images/893d36e42bf5ce55ea221b43653adbbc91539f0bc7c309089009a45a7e3fb853.jpg)  
(b)  $L = 0.03$

![](images/3d7622856d905e902adedd08171b188e8d6c133c15e96fe28dc9b7fb82555a54.jpg)  
Figure 20: LipNet under Model 2 with various L's.  
(c)  $L = 1$

![](images/8191765607345083b7840b62fe3652f3da9cfcdcf41be02f0cb1845ed74c76dc.jpg)  
(d)  $L = 3$

![](images/140bed43f3a969ce5442d745112f9b6e6151aa3750680c413ded0d57920e5599.jpg)

![](images/8d4f5a11f25cfb026fe2e5111569d53bbb2cc269d85141712a4dc9ede299978a.jpg)

![](images/bdb09c6ddc221faf98cddd74ed9625fce8d3771a30b66558e6bdad9bc41cc66a.jpg)

![](images/75393117659f8d4b66e042fe6158dc56ecaa26449c23ba48ff846ae6385d1c41.jpg)

![](images/cd5ab0ae7deaaca2ba44a787e0d638db09c44df5de90d73b833369c71a2da169.jpg)  
(a)  $\tau = 10^{-1}$  
Figure 21: LipNet under Model 2 with various  $\tau$ 's.

![](images/ea993c49db7a115a962613b1c9936e7ce4519e2c48be699b4a777a3c843955b7.jpg)  
(b)  $\tau = 10^{-2}$

![](images/6aacd4db5412f4a081354e7f9251d05dd683434bbc0301bfdbf6a13217f4c129.jpg)  
(c)  $\tau = 10^{-5}$

![](images/426489bcdfaf98ab73227474dd9f305753b3fe94e19895f32be120663874ff24.jpg)  
$(d)$ $\tau = 10^{-6}$

Table 4 Hyper-parameters  

<table><tr><td>Hyper-parameters</td><td>Configuration</td><td>Note</td></tr><tr><td>Sample size</td><td>1e4 for Model 1 &amp; 2, 1e6 for Model 3</td><td></td></tr><tr><td>k</td><td>100 for Model 1 &amp; 2, 300 for Model 3</td><td>See Definition 9</td></tr><tr><td rowspan="2">Network stucture</td><td>Layer-wise residual connection He et al. (2016), StdNet: batch normalization (Ioffe and Szegedy (2015)) after affine transformation</td><td></td></tr><tr><td>LipNet: Layer-wise residual connection (He et al. (2016)) with convex potential layer (Meunier et al. (2022))</td><td></td></tr><tr><td>Input dimension</td><td>dX</td><td></td></tr><tr><td>Output dimension</td><td>dY × Natom</td><td>Natom = k, see (10)</td></tr><tr><td>Number of hidden layers</td><td>5</td><td></td></tr><tr><td>Number of neurons each hidden layer</td><td>2k</td><td>k as in Definition 9</td></tr><tr><td>Activation function</td><td>StdNet: ReLU LipNet: ELU</td><td>See Section 5.3.1</td></tr><tr><td>L</td><td>0.1</td><td>See (18)</td></tr><tr><td>τ</td><td>1e-3</td><td>See Algorithm 2</td></tr><tr><td>Optimizer</td><td>Adam (Kingma and Ba (2017)) with learning rate 10-3</td><td>Learning rate is 10-2for StdNetin Model 1 &amp; 2</td></tr><tr><td>Batch size</td><td>100 for Model 1 &amp; 2256 for Model 3</td><td></td></tr><tr><td>Number of episodes</td><td>5e3 for Model 1 &amp; 2, 1e4 for Model 3</td><td></td></tr><tr><td>RBSP setting</td><td>25 partition, 8 query points each part</td><td>See Section 3.1.1</td></tr><tr><td>Random bisecting ratio</td><td>~ Uniform([0.45, 0.55])</td><td>See Section 3.1.1and Algorithm 3</td></tr><tr><td>Ratio for mandatory slicing along the longest edge</td><td>5</td><td>See Section 3.1.1and redge in Algorithm 3</td></tr><tr><td>Number of Sinkhorn iterations</td><td>5, if epoch ≤ 50010, if epoch &gt; 500</td><td></td></tr><tr><td>ε</td><td>1, if epoch ≤ 1000.1, if epoch ∈ [100, 500]0.05, if epoch &gt; 500</td><td>See (33)</td></tr><tr><td>Enforced sparsity</td><td>Off, if epoch ≤ 500On, if epoch &gt; 500</td><td>See Section 5.2</td></tr><tr><td>γ</td><td>0.5</td><td>See (34)</td></tr></table>

Table 5 Computational times of 500 training steps for LipNet, with  $M = 10^{6}$ .  

<table><tr><td>k</td><td>d=3</td><td>d=10</td></tr><tr><td>30</td><td>(24.6, 12.1)</td><td>(55.7, 16.3)</td></tr><tr><td>100</td><td>(27.0, 15.9)</td><td>(56.2, 16.9)</td></tr><tr><td>300</td><td>(29.2, 18.5)</td><td>(63.7, 24.7)</td></tr><tr><td>1000</td><td>(75.9, 64.8)</td><td>(151.3, 134.2)</td></tr></table>

This table presents the training time of LipNet for varying values of  $k$  and  $d_{\mathbb{X}} = d_{\mathbb{Y}} = d$ . For other hyper-parameters, please refer to Table 5. All times are reported in seconds. The first entry in each cell corresponds to the training time when using exact nearest neighbor search, while the second entry represents the time when employing ANNS-RBSP. The experiments were conducted on a machine equipped with an Nvidia L40 GPU.

# Appendix D. Another set of results on fluctuation

# D.1 On  $r$ -box estimator

Theorem 22 Under Assumptions 2 and 3, and choosing  $r$  as in Theorem 7, let  $\nu \in \mathcal{P}(\mathbb{X})$  be dominated by  $\lambda_{\mathbb{X}}$  with constant  $\overline{C} > 0$ . Then, there is a constant  $C > 0$  (which depends only on  $d_{\mathbb{X}}, \underline{c}, \overline{C}$  and the constants involved in  $r$ ), such that, for any  $\varepsilon \geq 0$ , we have

$$
\mathbb {P} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \hat {P} _ {x} ^ {r}\right) \mathrm {d} \nu (x) \geq \mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \hat {P} _ {x} ^ {r}\right) \mathrm {d} \nu (x) \right] + \varepsilon \right] \leq \left\{ \begin{array}{l l} \exp \left(- C M ^ {\frac {2}{d _ {\mathbb {X}} + 2}} \varepsilon^ {2}\right), & d _ {\mathbb {Y}} = 1, 2, \\ \exp \left(- C M ^ {\frac {d _ {\mathbb {X}}}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}} \varepsilon^ {2}}\right), & d _ {\mathbb {Y}} \geq 3. \end{array} \right.
$$

Proof Let  $\nu \in \mathcal{P}(\mathbb{X})$  as in the statement of the Theorem. We define

$$
Z := \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \hat {P} _ {x} ^ {r}\right) \mathrm {d} \nu (x),
$$

and introduce the following discrete time filtration:  $\mathcal{F}_0\coloneqq \{\emptyset ,\Omega \}$  and  $\mathcal{F}_m\coloneqq \sigma (\bigcup_{i = 1}^m\sigma (X_i,Y_i))$  for  $m = 1,\ldots ,M$ . We consider the Doob's martingale  $Z_{m}\coloneqq \mathbb{E}\left[Z\mid \mathcal{F}_{m}\right]$ ,  $m = 1,\dots ,M$ . Note that  $Z_{M} = Z$ . We will apply Azuma-Hoeffding inequality (cf. (Wainwright, 2019, Corollary 2.20)) to complete the proof.

Let us define

$$
\mathcal {D} ^ {m} := \left\{\left(X _ {1}, Y _ {1}\right), \dots , \left(X _ {m}, Y _ {m}\right), \left(x _ {m + 1}, y _ {m + 1}\right), \dots , \left(x _ {M}, y _ {M}\right) \right\}, \quad m = 1, \dots , M, \tag {35}
$$

$\mathcal{D}^0 \coloneqq \{(x_\ell, y_\ell)\}_{\ell=0}^M$ , and  $\mathcal{D}^M \coloneqq \mathcal{D}$ . Note that, for all  $m < M$ , we have, by Assumptions 3 (i), conditional Fubini-Tonelli theorem, and independent lemma,

$$
Z _ {m} = \int_ {\mathbb {X}} \int_ {(\mathbb {X} \times \mathbb {Y}) ^ {M - m}} \mathcal {W} \left(P _ {x}, \hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D} ^ {m}}\right) \bigotimes_ {\ell = m + 1} ^ {M} \psi (\mathrm {d} x _ {\ell} \mathrm {d} y _ {\ell}) \nu (\mathrm {d} x).
$$

This together with the linearity of integral, the fact that  $\psi$  is a probability, and the triangular inequality of  $\mathcal{W}$  implies that for  $m = 1,\ldots ,M$ ,

$$
\left| Z _ {m} - Z _ {m - 1} \right| \leq \int_ {\mathbb {X}} \int_ {\left(\mathbb {X} \times \mathbb {Y}\right) ^ {M - m + 1}} \mathcal {W} \left(\hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D} ^ {m}}, \hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D} ^ {m - 1}}\right) \bigotimes_ {\ell = m} ^ {M} \psi (\mathrm {d} x _ {\ell} \mathrm {d} y _ {\ell}) \nu (\mathrm {d} x). \tag {36}
$$

Notice that, by definitions (1) and (35),

$$
\left\{\hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathrm {D} _ {m}} \neq \hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathrm {D} _ {m - 1}} \right\} \subseteq \left\{X _ {m} \in \mathcal {B} ^ {r} (x) \right\} \cup \left\{x _ {m} \in \mathcal {B} ^ {r} (x) \right\}. \tag {37}
$$

Additionally, by definitions (1) and (35) again, on the event that  $\left\{\hat{\mu}_{\mathcal{B}^r(x)}^{\mathsf{D}_m} \neq \hat{\mu}_{\mathcal{B}^r(x)}^{\mathsf{D}_{m-1}}\right\}$ , we have

$$
\mathcal {W} \left(\hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D} ^ {m}}, \hat {\mu} _ {\mathcal {B} ^ {r} (x)} ^ {\mathcal {D} ^ {m - 1}}\right) \leq \left(1 + \sum_ {\ell = 1} ^ {m - 1} \mathbb {1} _ {\mathcal {B} ^ {r} (x)} \left(X _ {\ell}\right) + \sum_ {\ell = m + 1} ^ {M} \mathbb {1} _ {\mathcal {B} ^ {r} (x)} \left(x _ {\ell}\right)\right) ^ {- 1} \leq \left(1 + \sum_ {\ell = m + 1} ^ {M} \mathbb {1} _ {\mathcal {B} ^ {r} (x)} \left(x _ {\ell}\right)\right) ^ {- 1} (3 8)
$$

Combining (36), (37), (38), and Fubini-Tonelli theorem, we get

$$
\begin{array}{l} |Z_{m} - Z_{m - 1}|\leq \int_{\mathbb{X}}\int_{B(X_{m},2r)\cup B(x_{m},2r)}\int_{\mathbb{X}^{M + 1 - m}}\left(1 + \sum_{\ell = m + 1}^{M}\mathbb{1}_{\mathcal{B}^{r}(x)}(x_{\ell})\right)^{-1}\bigotimes_{\ell = m + 1}^{M}\xi (\mathrm{d}x_{\ell})\nu (\mathrm{d}x)\xi (\mathrm{d}x_{m}) \\ \leq \sup  _ {x _ {m} \in \mathbb {X}} \int_ {B \left(X _ {m}, 2 r\right) \cup B \left(x _ {m}, 2 r\right)} \int_ {\mathbb {X} ^ {M + 1 - m}} \left(1 + \sum_ {\ell = m + 1} ^ {M} \mathbb {1} _ {\mathcal {B} ^ {r} (x)} \left(x _ {\ell}\right)\right) ^ {- 1} \bigotimes_ {\ell = m + 1} ^ {M} \xi (\mathrm {d} x _ {\ell}) \nu (\mathrm {d} x) \tag {39} \\ \end{array}
$$

where the  $2r$  in the domain of the integral stems from the usage of  $\beta^r$  in the definition of  $\mathcal{B}^r$  (see Definition 5). Now, for fixed  $x, x_m \in \mathbb{X}$ , we have

$$
\begin{array}{l} \int_ {\mathbb {X} ^ {M - m + 1}} \left(1 + \sum_ {\ell = m + 1} ^ {M} \mathbb {1} _ {\mathcal {B} ^ {r} (x)} (x _ {\ell})\right) ^ {- 1} \bigotimes_ {\ell = m + 1} ^ {M} \xi (\mathrm {d} x _ {\ell}) \\ = \sum_ {\ell = 0} ^ {M - m} \binom {M - m} {\ell} \xi \left(\mathcal {B} ^ {r} (x)\right) ^ {\ell} \left(1 - \xi^ {r} \left(\mathcal {B} ^ {r} (x)\right)\right) ^ {M - m - \ell} (1 + \ell) ^ {- 1} \\ = \frac{1}{(M - m + 1)\xi\big(\mathcal{B}^{r}(x)\big)}\sum_{\ell = 0}^{M - m}\binom {M - m + 1}{\ell + 1}\xi \big(\mathcal{B}^{r}(x)\big)^{\ell +1}\Bigl(1 - \xi \big(\mathcal{B}^{r}(x)\big)\Bigr)^{M - m - \ell} \\ = \frac {1}{(M - m + 1) \xi \big (\mathcal {B} ^ {r} (x) \big)} \sum_ {\ell = 1} ^ {M - m + 1} \binom {M - m + 1} {\ell} \xi \big (\mathcal {B} ^ {r} (x) \big) ^ {\ell} \Big (1 - \xi \big (\mathcal {B} ^ {r} (x) \big) \Big) ^ {M - m + 1 - \ell} \\ = \frac {1 - \left(1 - \xi \big (\mathcal {B} ^ {r} (x) \big)\right) ^ {M - m + 1}}{(M - m + 1) \xi \big (\mathcal {B} ^ {r} (x) \big)} \leq 1 \wedge \big ((M - m + 1) \xi \big (\mathcal {B} ^ {r} (x) \big) \big) ^ {- 1} \leq 1 \wedge \Big ((M - m + 1) \underline {{c}} (2 r) ^ {d _ {\mathbb {X}}} \Big) ^ {- 1}, \\ \end{array}
$$

where we have used Assumption 3 (ii) in the last inequality. Recall  $\overline{C}$  introduced in Theorem 22. In view of (39), we have

$$
\begin{array}{l} \left| Z _ {m} - Z _ {m - 1} \right| \leq \sup  _ {x _ {m} \in \mathbb {X}} \int_ {B (X _ {m}, 2 r) \cup B (x _ {m}, 2 r)} 1 \wedge \left(\left(M - m + 1\right) \underline {{c}} (2 r) ^ {d _ {\mathbb {X}}}\right) ^ {- 1} \nu (\mathrm {d} x) \\ \leq 2 \bar {C} (4 r) ^ {d _ {\mathbb {X}}} \left(1 \wedge \left(\left(M - m + 1\right) \underline {{c}} (2 r) ^ {d _ {\mathbb {X}}}\right) ^ {- 1}\right) \\ = \left(\overline {{C}} 2 ^ {2 d _ {\mathbb {X}} + 1} r ^ {d _ {\mathbb {X}}}\right) \wedge \frac {\overline {{C}} 2 ^ {d _ {\mathbb {X}} + 1}}{\underline {{c}} (M - m + 1)} := C _ {m}. \\ \end{array}
$$

By Azuma-Hoeffding inequality (cf. (Wainwright, 2019, Corollary 2.20)), one obtains

$$
\mathbb {P} (Z - \mathbb {E} [ Z ] \geq \varepsilon) \leq \exp \left(- \frac {2 \varepsilon^ {2}}{\sum_ {m = 1} ^ {M} C _ {m} ^ {2}}\right).
$$

To complete the proof, we substitute in the configuration of Theorem 7. Since we only aim to investigate the rate of  $\sum_{m=1}^{M} C_m^2$  as  $M \to \infty$ , we simply set

$$
r = M ^ {- \frac {1}{d _ {\mathbb {X}} + d}} \quad \text {w i t h} \quad d := 2 \lor d _ {\mathbb {Y}}.
$$

It follows that

$$
\begin{array}{l} \sum_ {m = 1} ^ {M} C _ {m} ^ {2} \sim \sum_ {m = 1} ^ {M} M ^ {- \frac {2 d _ {\mathbb {X}}}{d _ {\mathbb {X}} + d}} \wedge m ^ {- 2} \\ \lesssim \int_ {1} ^ {\infty} M ^ {- \frac {2 d _ {\mathbb {X}}}{d _ {\mathbb {X}} + d}} \wedge z ^ {- 2} \mathrm {d} z \sim \int_ {1} ^ {M ^ {\frac {d _ {\mathbb {X}}}{d _ {\mathbb {X}} + d}}} M ^ {- \frac {2 d _ {\mathbb {X}}}{d _ {\mathbb {X}} + d}} \mathrm {d} z + \int_ {M ^ {\frac {d _ {\mathbb {X}}}{d _ {\mathbb {X}} + d}}} ^ {\infty} z ^ {- 2} \mathrm {d} z \sim M ^ {- \frac {d _ {\mathbb {X}}}{d _ {\mathbb {X}} + d}}, \\ \end{array}
$$

which completes the proof.

# D.2 On  $k$ -nearest-neighbor estimator

Theorem 23 Under Assumptions 2 and 3, and the choice of  $k$  as in Theorem 10, there is a constant  $C > 0$  (which depends only on  $\underline{c}$  and the constants involved in  $k$ ), such that, for any  $\nu \in \mathcal{P}(\mathbb{X})$  and  $\varepsilon \geq 0$ , we have

$$
\begin{array}{r} \mathbb {P} \left[ \int_ {\mathbb {X}} \mathcal {W} (P _ {x}, \check {P} _ {x} ^ {k}) \nu (\mathrm {d} x) \geq \mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} (P _ {x}, \check {P} _ {x} ^ {k}) \nu (\mathrm {d} x) \right] + \varepsilon \right] \leq \left\{ \begin{array}{l l} \exp (- C M ^ {\frac {2}{d _ {\mathbb {X}} + 2}} \varepsilon^ {2}), & d _ {\mathbb {Y}} = 1, 2, \\ \exp (- C M ^ {\frac {d _ {\mathbb {Y}}}{d _ {\mathbb {X}} + d _ {\mathbb {Y}}}} \varepsilon^ {2}), & d _ {\mathbb {Y}} \geq 3. \end{array} \right. \end{array}
$$

Proof [Proof of Theorem 23] For notational convenience, we will write  $\hat{\mu}_{\mathcal{N}^k (x)}^{\mathrm{D}}$  for  $\hat{\mu}_{\mathcal{N}^k,\mathsf{D}(x)}^{\mathrm{D}}$ . Clearly, with  $\mathsf{D} = \mathcal{D}$ , we recover  $\hat{\mu}_{\mathcal{N}^k (x)}^{\mathcal{D}} = \hat{\mu}_{\mathcal{N}^k,\mathcal{D}(x)}^{\mathcal{D}} = \check{P}_x^k$ . In what follows, we let

$$
Z := \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \check {P} _ {x} ^ {k}\right) \nu (\mathrm {d} x).
$$

We also define  $\mathcal{F}_0\coloneqq \{\emptyset ,\Omega \}$  and  $\mathcal{F}_m\coloneqq \sigma (\bigcup_{i = 1}^m\sigma (X_i,Y_i))$  for  $m = 1,\ldots ,M$ . The proof relies on an application of Azuma-Hoeffding inequality (cf. (Wainwright, 2019, Corollary 2.20)) to the Doob's martingale  $\{\mathbb{E}[Z|\mathcal{F}_m]\}_{m = 0}^M$ . In order to proceed, we introduce a few more notations:

$$
\boldsymbol {x} := \left(x _ {1}, \dots , x _ {M}\right), \quad \mathcal {X} := \left(X _ {1}, \dots , X _ {M}\right),
$$

$$
\mathcal {X} ^ {m} := \left(X _ {1}, \dots , X _ {m}, x _ {m + 1}, \dots , x _ {M}\right),
$$

$$
\mathcal {D} ^ {m} := \left\{\left(X _ {1}, Y _ {1}\right), \ldots , \left(X _ {m}, Y _ {m}\right), \left(x _ {m + 1}, y _ {m + 1}\right), \ldots , \left(x _ {M}, y _ {M}\right) \right\},
$$

$$
\eta_ {x} ^ {k, \boldsymbol {x}} := \text {t h e} k \text {- t h s m a l l e s t o f} \left\{\left\| x _ {m} - x \right\| _ {\infty} \right\} _ {m = 1} ^ {M}.
$$

By independence lemma, we have

$$
\begin{array}{l} \mathbb {E} \left[ Z \big | \mathcal {F} _ {m} \right] = \int_ {(\mathbb {X} \times \mathbb {Y}) ^ {M - m}} \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \hat {\mu} _ {\mathcal {N} ^ {k} (x)} ^ {\mathcal {D} ^ {m}}\right) \nu (\mathrm {d} x) \bigotimes_ {\ell = m + 1} ^ {M} \psi (\mathrm {d} x _ {\ell} \mathrm {d} y _ {\ell}) \\ = \int_ {(\mathbb {X} \times \mathbb {Y}) ^ {M - m}} \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \frac {1}{k} \left(\sum_ {i = 1} ^ {m} \mathbb {1} _ {\| X _ {i} - x \| _ {\infty} \leq \eta_ {x} ^ {k, \mathcal {X} ^ {m}}} \delta_ {Y _ {i}} + \sum_ {\ell = m + 1} ^ {M} \mathbb {1} _ {\| x _ {\ell} - x \| _ {\infty} \leq \eta_ {x} ^ {k, \mathcal {X} ^ {m}} \delta_ {y _ {\ell}}}\right)\right) \nu (\mathrm {d} x) \\ \end{array}
$$

$$
\bigotimes_ {\ell = m + 1} ^ {M} \psi (\mathrm {d} x _ {\ell} \mathrm {d} y _ {\ell}),
$$

where we note that  $(\mathbb{X}\times \mathbb{Y})^{M - m}$  and  $\bigotimes_{\ell = m + 1}^{M}\psi (\mathrm{d}x_{\ell}\mathrm{d}y_{\ell})$  in the right hand side can be replaced by  $(\mathbb{X}\times \mathbb{Y})^{M - m + 1}$  and  $\bigotimes_{\ell = m}^{M}\psi (\mathrm{d}x_{\ell}\mathrm{d}y_{\ell})$  as the integrand is constant in  $x_{m}$  and  $\psi$  is a probability measure. Therefore, by Fubini's theorem and triangle inequality for  $\mathcal{W}$ , we have

$$
\begin{array}{l} \left| \mathbb {E} \left[ Z \middle | \mathcal {F} _ {m} \right] - \mathbb {E} \left[ Z \middle | \mathcal {F} _ {m - 1} \right] \right| \\ \leq \int_ {\mathbb {X}} \int_ {(\mathbb {X} \times \mathbb {Y}) ^ {M - m + 1}} \mathcal {W} \left(\frac {1}{k} \left(\sum_ {i = 1} ^ {m} \mathbb {1} _ {\| X _ {i} - x \| _ {\infty} \leq \eta_ {x} ^ {k, \chi^ {m}}} \delta_ {Y _ {i}} + \sum_ {\ell = m + 1} ^ {M} \mathbb {1} _ {\| x _ {\ell} - x \| _ {\infty} \leq \eta_ {x} ^ {k, \chi^ {m}} \delta_ {y _ {\ell}}}\right) \right. \\ \frac {1}{k} \left(\sum_ {i = 1} ^ {m - 1} \mathbb {1} _ {\| X _ {i} - x \| _ {\infty} \leq \eta_ {x} ^ {k, \chi^ {m - 1}} \delta Y _ {i}} + \sum_ {\ell = m} ^ {M} \mathbb {1} _ {\| x _ {\ell} - x \| _ {\infty} \leq \eta_ {x} ^ {k, \chi^ {m - 1}} \delta y _ {\ell}}\right) \biggr) \bigotimes_ {\ell = m} ^ {M} \psi (\mathrm {d} x _ {\ell} \mathrm {d} y _ {\ell}) \mathrm {d} x. \tag {40} \\ \end{array}
$$

Above, the only difference between the two measures inside  $\mathcal{W}$  is the  $m$ -th summand. Due to the definition of  $\mathcal{W}$  and the boundedness of  $\mathbb{X}$ , the transport cost induced by altering the  $m$ -th summand is at most  $k^{-1}$ . It follows that

$$
\left| \mathbb {E} \left[ Z \mid \mathcal {F} _ {m} \right] - \mathbb {E} \left[ Z \mid \mathcal {F} _ {m - 1} \right] \right| \leq \frac {1}{k}, \quad m = 1, \dots , M. \tag {41}
$$

Below we further refine the upper bound of the absolute difference in the left hand side of (40) when  $m = 1, \ldots, M - k$ . For the integrand in the right hand side of (40) to be positive, it is necessary that

$$
\mathbb {1} _ {\| X _ {m} - x \| _ {\infty} \leq \eta_ {x} ^ {k, \mathcal {X} ^ {m}}} + \mathbb {1} _ {\| x _ {m} - x \| _ {\infty} \leq \eta_ {x} ^ {k, \mathcal {X} ^ {m - 1}}} \geq 1.
$$

This, together with the tie breaking rule stipulated in Definition 9, further implies that

$$
\mathbb {1} _ {A _ {1} ^ {m}} + \mathbb {1} _ {A _ {2} ^ {m}} \geq 1,
$$

where

$$
\begin{array}{l} A _ {1} ^ {m} := \left\{\mathrm {a t m o s t} (k - 1) \mathrm {o f} x _ {\ell}, \ell = m + 1, \ldots , M - m, \mathrm {f a l l s i n t o} B _ {x} ^ {\| X _ {m} - x \| _ {\infty}} \right\}, \\ A _ {2} ^ {m} := \left\{\mathrm {a t m o s t} (k - 1) \mathrm {o f} x _ {\ell}, \ell = m + 1, \ldots , M - m, \mathrm {f a l l s i n t o} B _ {x} ^ {\| x _ {m} - x \| _ {\infty}} \right\}. \\ \end{array}
$$

Combining the above with the reasoning leading to (41), we yield

$$
\begin{array}{l} \left| \mathbb {E} \left[ Z \middle | \mathcal {F} _ {m} \right] - \mathbb {E} \left[ Z \middle | \mathcal {F} _ {m - 1} \right] \right| \\ \leq \frac {1}{k} \left(\int_ {\mathbb {X}} \int_ {(\mathbb {X} \times \mathbb {Y}) ^ {M - m}} \mathbb {1} _ {A _ {1} ^ {m}} \bigotimes_ {\ell = m + 1} ^ {M} \xi (\mathrm {d} x _ {\ell}) \nu (\mathrm {d} x) + \int_ {\mathbb {X}} \int_ {(\mathbb {X} \times \mathbb {Y}) ^ {M - m + 1}} \mathbb {1} _ {A _ {2} ^ {m}} \bigotimes_ {\ell = m} ^ {M} \xi (\mathrm {d} x _ {\ell}) \nu (\mathrm {d} x)\right) \\ \end{array}
$$

Above, we have replaced  $\psi$  in (40) by  $\xi$  because  $A_1^m$  and  $A_2^m$  no longer depend on  $y_{\ell}, \ell = m + 1, \ldots, M$ . The analogue applies to the domain of integral as well. We continue to have

$$
\begin{array}{l} \left| \mathbb {E} \left[ Z \mid \mathcal {F} _ {m} \right] - \mathbb {E} \left[ Z \mid \mathcal {F} _ {m - 1} \right] \right| \\ \leq \frac {1}{k} \left(\int_ {\mathbb {X}} \int_ {(\mathbb {X} \times \mathbb {Y}) ^ {M - m}} \mathbb {1} _ {A _ {1} ^ {m}} \bigotimes_ {\ell = m + 1} ^ {M} \xi (\mathrm {d} x _ {\ell}) \nu (\mathrm {d} x) + \int_ {\mathbb {X}} \int_ {(\mathbb {X} \times \mathbb {Y}) ^ {M - m + 1}} \mathbb {1} _ {A _ {2} ^ {m}} \bigotimes_ {\ell = m} ^ {M} \xi (\mathrm {d} x _ {\ell}) \nu (\mathrm {d} x)\right) \tag {42} \\ =: \frac {1}{k} \left(I _ {1} ^ {m} + I _ {2} ^ {m}\right). \\ \end{array}
$$

Regarding  $I_{m}^{1}$  defined in (42), note that by Assumption 3,

$$
\int_ {(\mathbb {X} \times \mathbb {Y}) ^ {M - m}} \mathbb {1} _ {A _ {1} ^ {m}} \bigotimes_ {\ell = m + 1} ^ {M} \xi (\mathrm {d} x _ {\ell}) = \mathbb {P} \left[ \mathrm {a t m o s t} (k - 1) \mathrm {o f} \check {X} _ {1}, \ldots , \check {X} _ {M - m} \mathrm {f a l l s i n t o} B _ {x} ^ {\| x ^ {\prime} - x \| _ {\infty}} \right] \Big | _ {x ^ {\prime} = X _ {m}},
$$

where  $\check{X}_1, \ldots, \check{X}_{M - m} \stackrel{\mathrm{i.i.d.}}{\sim} \xi$ . Below we define a CDF  $G(r) \coloneqq \underline{c} r^d, r \in [0, \underline{c}^{-\frac{1}{d}}]$ . By Assumption 3 (ii), for any  $x, x' \in \mathbb{X}$ , we have

$$
\int_ {\mathbb {X}} \mathbb {1} _ {\check {x} \in B _ {x} ^ {\| x ^ {\prime} - x \| _ {\infty}}} \xi (\mathrm {d} \check {x}) \geq \underline {{c}} \int_ {\mathbb {X}} \mathbb {1} _ {\check {x} \in B _ {x} ^ {\| x ^ {\prime} - x \| _ {\infty}}} \mathrm {d} \check {x} \geq \underline {{c}} \int_ {\mathbb {X}} \mathbb {1} _ {\check {x} \in B _ {\mathbf {0}} ^ {\| x ^ {\prime} - x \| _ {\infty}}} \mathrm {d} \check {x} = G (\| x ^ {\prime} - x \| _ {\infty}),
$$

where we have used the fact that  $\| x' - x\|_{\infty} \leq 1 \leq \underline{c}^{-\frac{1}{d}}$  in the last equality. It follows from Lemma 17 that

$$
\int_{(\mathbb{X}\times \mathbb{Y})^{M - m}}\mathbb{1}_{A_{1}^{m}}\bigotimes_{\ell = m + 1}^{M}\xi (\mathrm{d}x_{\ell})\\ \leq \sum_{j = 0}^{k - 1}\binom {M - m}{j}G(\| X_{m} - x\|_{\infty})^{j}\big(1 - G(\| X_{m} - x\|_{\infty})\big)^{M - m - j},
$$

and thus, by letting  $U\sim \mathrm{Uniform}(\mathbb{X})$

$$
\begin{array}{l} I_{1}^{m}\leq \int_{\mathbb{X}}\sum_{j = 0}^{k - 1}\binom {M - m}{j}G(\| X_{m} - x\|_{\infty})^{j}\big(1 - G(\| X_{m} - x\|_{\infty})\big)^{M - m - j}\mathrm{d}x \\ = \mathbb {E} \left[ \sum_ {j = 0} ^ {k - 1} {\binom {M - m} {j}} G (\| x ^ {\prime} - U \| _ {\infty}) ^ {j} \big (1 - G (\| x ^ {\prime} - U \| _ {\infty}) \big) ^ {M - m - j} \right] \Big | _ {x ^ {\prime} = X _ {m}}, \\ \end{array}
$$

where we note that the upper bounded no longer involves  $\nu$ . For  $x' \in \mathbb{X}$ , it is obvious that

$$
\mathbb {P} \big [ \| x ^ {\prime} - U \| _ {\infty} \leq r \big ] \geq \mathbb {P} \big [ \| U \| _ {\infty} \leq r \big ], \quad r \in \mathbb {R},
$$

i.e.,  $\| U\|_{\infty}$  stochastically dominates  $\| x^{\prime} - U\|_{\infty}$ . Note additionally that, by Lemma 17 again, below is a non-decreasing function,

$$
r\mapsto \sum_{j = 0}^{k - 1}\binom {M - m}{j}G(r)^{j}\big(1 - G(r)\big)^{M - m - j}.
$$

Consequently,

$$
I_{1}^{m}\leq \mathbb{E}\left[\sum_{j = 0}^{k - 1}\binom {M - m}{j}G(\| U\|_{\infty})^{j}\big(1 - G(\| U\|_{\infty})\big)^{M - m - j}\mathrm{d}x\right].
$$

Since  $\| U\|_{\infty}$  has CDF  $r\mapsto r^{d_{\mathbb{X}}},r\in [0,1]$  and  $G(r) = \underline{c} r^d,r\in [0,\underline{c}^{-\frac{1}{d}}]$ , we continue to obtain

$$
\begin{array}{l} I _ {1} ^ {m} \leq \sum_ {j = 0} ^ {k - 1} \binom {M - m} {j} \int_ {r = 0} ^ {1} \underline {{c}} r ^ {d _ {\mathbb {X}} j} (1 - \underline {{c}} r ^ {d _ {\mathbb {X}}}) ^ {M - m - j}   \mathrm {d} r ^ {d _ {\mathbb {X}}} \\ \leq \underline {{c}} ^ {- 1} \sum_ {j = 0} ^ {k - 1} \frac {(M - m) !}{j ! (M - m - j) !} \int_ {0} ^ {1} r ^ {j} (1 - r) ^ {M - m - j} \mathrm {d} r. \\ \end{array}
$$

With a similar calculation as in (26), which involves beta distribution and gamma function, we arrive at

$$
I _ {1} ^ {m} \leq \underline {{c}} \sum_ {j = 0} ^ {k - 1} \frac {(M - m) !}{j ! (M - m - j) !} \frac {j ! (M - m - j) !}{(M - m + 1) !} \leq \frac {\underline {{c}} ^ {- 1} k}{M - m}. \tag {43}
$$

Regarding  $I_2^m$  defined in (42), we first let  $\check{X}_0, \check{X}_1, \ldots, \check{X}_{M - m} \stackrel{\mathrm{i.i.d.}}{\sim} \xi$ . Then, note that

$$
\begin{array}{l} \int_ {(\mathbb {X} \times \mathbb {Y}) ^ {M - m}} \mathbb {1} _ {A _ {2} ^ {m}} \bigotimes_ {\ell = m + 1} ^ {M} \xi (\mathrm {d} x _ {\ell}) \leq \mathbb {P} \left[ \text {a t m o s t} (k - 1) \text {o f} \check {X} _ {1}, \dots , \check {X} _ {M - m} \text {f a l l s i n t o} B _ {x} ^ {\| \check {X} _ {0} - x \| _ {\infty}} \right] \\ \leq \binom {M - m - 1} {k - 1} \binom {M - m} {k} ^ {- 1} = \frac {k}{M - m}, \\ \end{array}
$$

where the inequality in the second line is due to the symmetry stemming from Assumption 3 (i), and the fact that congestion along with the tie-breaking rule specified in Definition 9 may potentially rules out certain permutations. Consequently,

$$
I _ {2} ^ {m} \leq \frac {k}{M - m}. \tag {44}
$$

Putting together (41), (42), (43), and (44), we yield

$$
\left| \mathbb {E} \left[ Z \middle | \mathcal {F} _ {m} \right] - \mathbb {E} \left[ Z \middle | \mathcal {F} _ {m - 1} \right] \right| \leq C _ {m} := \frac {\overline {{C}} (\underline {{c}} ^ {- 1} + 1)}{M - m} \wedge \frac {1}{k}, \quad m = 1, \ldots , M.
$$

By Azuma-Hoeffding inequality (cf. (Wainwright, 2019, Corollary 2.20)),

$$
\mathbb {P} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \hat {\mu} _ {\mathcal {N} ^ {k} (x)} ^ {\mathcal {D}}\right) \nu (\mathrm {d} x) - \mathbb {E} \left[ \int_ {\mathbb {X}} \mathcal {W} \left(P _ {x}, \hat {\mu} _ {\mathcal {N} ^ {k} (x)} ^ {\mathcal {D}}\right) \nu (\mathrm {d} x) \right] \geq \varepsilon \right] \leq \exp \left(- \frac {\varepsilon^ {2}}{2 \sum_ {m = 1} ^ {M} C _ {m} ^ {2}}\right), \varepsilon \geq 0.
$$

To complete the proof, we substitute in the configuration of Theorem 10. Below we only investigate the rate of  $\sum_{m=1}^{M} C_m^2$  as  $M \to \infty$ , and do not keep track of the constant. For simplicity, we set

$$
k = k \sim M ^ {\frac {d}{d _ {\mathbb {X}} + d}} \quad \mathrm {w i t h} \quad d := 2 \vee d _ {\mathbb {Y}}
$$

It follows that

$$
\sum_ {m = 1} ^ {M} C _ {m} ^ {2} \sim \sum_ {m = 1} ^ {\lfloor M - M ^ {\frac {d}{d _ {\mathbb {X}} + d}} \rfloor} \frac {1}{(M - m) ^ {2}} + \frac {M ^ {\frac {d}{d _ {\mathbb {X}} + d}}}{M ^ {\frac {2 d}{d _ {\mathbb {X}} + d}}} \sim \int_ {M} ^ {\infty} \frac {d}{d _ {\mathbb {X}} + d} \frac {1}{r ^ {2}} \mathrm {d} r + \frac {1}{M ^ {\frac {d}{d _ {\mathbb {X}} + d}}} \sim M ^ {- \frac {d}{d _ {\mathbb {X}} + d}},
$$

which completes the proof.

# References

B. Acciaio and S. Hou. Convergence of adapted empirical measures on  $\mathbb{R}^d$ . arXiv:2211.10162, 2023.  
B. Acciaio, A. Kratsios, and G. Pammer. Designing universal causal deep learning models: The geometric (hyper)transformer. Mathematical Finance, 34:671-735, 2024.  
C. D. Aliprantis and K. C. Border. Infinite Dimensional Analysis: A Hitchhiker's Guide. Springer-Verlag Berlin Heidelberg, 2006.  
F. Altekrüger, P. Hagemann, and G. Steidl. Conditional generative models are provably robust: Pointwise guarantees for bayesian inverse problems. Transactions on Machine Learning Research, 2023.  
A. Araujo, A. J. Havens, B. Delattre, A. Allauzen, and B. Hu. A unified algebraic perspective on lipschitz neural networks. The Eleventh International Conference on Learning Representations, 2023.  
J. Backhoff, D. Bartl, M. Beiglböck, and J. Wiesel. Estimating processes in adapted Wasserstein distance. Annals of Applied Probability, 32:529-550, 2022.  
T. Bai, J. Luo, Jun Zhao, B. Wen, and Qian Wang. Recent advances in adversarial training for adversarial robustness. Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, 2021.  
P. L. Bartlett, D. J. Foster, and M. J. Telgarsky. Spectrally-normalized margin bounds for neural networks. Advances in Neural Information Processing Systems, 30, 2017.  
D. M. Bashtannyk and R. J. Hyndman. Bandwidth selection for kernel conditional density estimation. Computational Statistics & Data Analysis, 36:279-298, 2001.  
K. Bertin, C. Lacour, and V. Rivoirard. Adaptive pointwise estimation of conditional density function. Ann. Inst. H. Poincaré Probab. Statist., 52:939-980, 2016.  
P. K. Bhattacharya and A. K. Gangopadhyay. Kernel and nearest-neighbor estimation of a conditional quantile. The Annals of Statistics, 18:1400-1415, 1990.  
A. Bhowmick, M. D'Souza, and G. S. Raghavan. Lipbab: Computing exact lipschitz constant of relu networks. International Conference on Artificial Neural Networks, 2021.  
G. Biau and L. Devroye. Lectures on the Nearest Neighbor Method. Springer, 2015.  
V. I. Bogachev. Measure Theory Volume II. Springer-Verlag Berlin Heidelberg, 2007.  
J. Booth, P. Hall, and A. Wood. Bootstrap estimation of conditional distributions. The Annals of Statistics, 20:1594-1610, 1992.  
P. Bountakas, A. Zarras, A. Lekidis, and C. Xenakis. Defense strategies for adversarial machine learning: A survey. Computer Science Review, 49, 2023.  
Z. Cheng and S. Jaimungal. Distributional dynamic risk measures in markov decision processes. arXiv:2203.09612, 2023.  
Y. Chow, A. Tamar, S. Mannor, and M. Pavone. Risk-sensitive and robust decision-making: a cvar optimization approach. Advances in Neural Processing Systems, 2015.

D. Clevert, T. Unterthiner, and S. Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). arXiv:1511.07289, 2016.  
Anthony Coache, Sebastian Jaimungal, and Álvaro Cartea. Conditionally elicitable dynamic risk measures for deep reinforcement learning. SIAM Journal on Financial Mathematics, 14(4):1249-1289, 2023.  
J. Cohen, E. Rosenfeld, and Z. Kolter. Certified adversarial robustness via randomized smoothing. Proceedings of the 36th International Conference on Machine Learning, 97, 2019.  
E. Demirkayaa, Y. Fan, L. Gao, J. Lv, P. Vossler, and J. Wang. Optimal nonparametric inference with two-scale distributional nearest neighbors. Journal of the American Statistical Association, 199:297-307, 2024.  
L. Devroye. Necessary and sufficient conditions for the pointwise convergence of nearest neighbor regression function estimates. Probability Theory and Related Fields, 61:467-481, 1982.  
R. M. Dudley. The speed of mean glivenko-cantelli convergence. The Annals of Mathematical Statistics, 40(1):40-50, 1969.  
J. Fan. Design-adaptive nonparametric regression. Journal of the American Statistical Association, 87:998-1004, 1992.  
M. Fazlyab, A. Robey, H. Hassani, M. Morari, and G. Pappas. Efficient and accurate estimation of lipschitz constants for deep neural networks. Advances in Neural Information Processing Systems, 32, 2019.  
M. Fazlyab, T. Entesari, A. Roy, and R. Chellappa. Certified robustness via dynamic margin maximization and improved lipschitz regularization. Advances in Neural Information Processing Systems, 36, 2024.  
F. Ferraty and P. Vieu. Nonparametric Functional Data Analysis: Theory and Practices. Springer, 2006.  
E. Fetaya, J., W. Grathwohl, and R. Zemel. Understanding the limitations of conditional generative models. International Conference on Learning Representations, 2020.  
E. Fix and J. L. Hodges. Discriminatory analysis; nonparametric discrimination, consistency properties. USAF SAM Series in Statistics, Project No. 21-49-004, 1951.  
N. Fournier. Convergence of the empirical measure in expected wasserstein distance: non-asymptotic explicit bounds in  $\mathbb{R}^d$ . ESAIM: Probability and Statistics, 27:749-775, 2023.  
Nicolas Fournier and Arnaud Guillin. On the rate of convergence in wasserstein distance of the empirical measure. Probability Theory and Related Fields, 162:707-738, 2015.  
T. Gasser and H. Müller. Kernel estimation of regression functions. Smoothing Technique for Curve Estimation, pages 23-63, 1979.  
M. Giegrich, R. Oomen, and C. Reisinger. K-nearest-neighbor resampling for off-policy evaluation in stochastic control. arXiv:2306.04836, 2024.  
I. Gijbels and A. Goderniaux. Bandwidth selection for changepoint estimation in nonparametric regression. Technometrics, 46:76-86, 2004.

H. Gouk, E. Frank, B. Pfahringer, and M. J. Cree. Regularisation of neural networks by enforcing lipschitz continuity. Machine Learning, 110:393-416, 2021.  
L. Györfi, M. Kohler, A. Krzyzak, and H. Walk. A Distribution-Free Theory of Nonparametric Regression. Springer, 2002.  
K. Hajebi, Y. Abbasi-Yadkori, H. Shahbazi, and H. Zhang. Fast approximate nearest-neighbor search with k-nearest neighbor graph. Proceedings of the Twenty-Second International Joint Conference on Artificial Intelligence, 2011.  
P. Hall and J. D. Hart. Nonparametric regression with long-range dependence. Stochastic Processes and their Applications, 36:339-351, 1990.  
P. Hall, R. C. L. Wolff, and Q. Yao. Methods for estimating a conditional distribution function. Journal of the American Statistical Association, 94:154-163, 1999.  
W. Hardle and J. S. Marron. Optimal bandwidth selection in nonparametric regression function estimation. The Annals of Statistics, 13:1465-1481, 1985.  
K. He, X. Zhang, S., and J. Sun. Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770-778, 2016.  
Ruiyang Hong and Anastasis Kratsios. Bridging the gap between approximation and learning via optimal approximation by relu mlps of maximal regularity. arXiv:2409.12335, 2024.  
B. Hosseini, A. W. Hsu, and A. Taghvaei. Conditional optimal transport on function spaces. arXiv:2311.05672, 2024.  
S. Hou. Convergence of the adapted smoothed empirical measures. arXiv:2401.14883, 2024.  
W. Huang and W. B. Haskell. Risk-aware q-learning for markov decision processes. IEEE 56th Annual Conference on Decision and Control, 2017.  
Y. Huang, H. Zhang, Y. Shi, J. Z. Kolter, and A. Anandkumar. Training certifiably robust neural networks with efficient local lipschitz bounds. Advances in Neural Information Processing Systems, 34, 2021.  
S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. Proceedings of the 32nd International Conference on Machine Learning, pages 448-456, 2015.  
M. Jordan and A. G. Dimakis. Exactly computing the local lipschitz constant of relu networks. Advances in Neural Information Processing Systems, 33, 2021.  
D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv:1412.6980, 2017.  
Benoit R. Kloeckner. Empirical measures: regularity is a counter-curse to dimensionality. ESAIM: Probability and Statistics, 24:408-434, 2020.  
M. Kohler, A. Krzyżak, and H. Walk. Optimal global rates of convergence for nonparametric regression with unbounded data. Journal of Statistical Planning and Inference, 193:1286-1296, 2009.

M. Kohler, A. Krzyzak, and H. Walk. Uniform convergence rate for nonparametric regressions and principle component analysis with functional/longitudinal data. The Annals of Statistics, 38: 3321-3351, 2010.  
M. Kohler, A. Schindler, and S. Sperlich. A review and comparison of bandwidth selection methods for kernel regression. International Statistical Review, 82:243-274, 2014.  
A. Kratsios. Universal regular conditional distributions via probabilistic transformers. Constructive Approximation, 57:1145-1212, 2023.  
C. Lacour. Adaptive pointwise estimation of conditional density function. Ann. Inst. H. Poincaré Probab. Statist., 43:571-597, 2007.  
Wen Li, Ying Zhang, Yifang Sun, Wei Wang, Mingjie Li, Wenjie Zhang, and Xuemin Lin. Approximate nearest neighbor search on high dimensional data — experiments, analyses, and improvement. IEEE Transactions on Knowledge and Data Engineering, 32:1475-1488, 2020a.  
Y. Li, S. Akbar, and J. Oliva. Acflow: Flow models for arbitrary conditional likelihoods. Proceedings of the 37th International Conference on Machine Learning, 119:5831-5841, 2020b.  
H. D. Liu, F. Williams, A. Jacobson, S. Fidler, and O. Litany. Learning smooth neural functions via lipschitz regularization. SIGGRAPH '22: Special Interest Group on Computer Graphics and Interactive Techniques Conference, 2022.  
Y. P. Mack. Local properties of  $k$ -nn regression estimates. SIAM Journal on Algebraic Discrete Methods, 2:311-323, 1981.  
F. Martínez, M. P. Friás, M. D. Pérez, and A. J. Rivera. A methodology for applying k-nearest neighbor to time series forecasting. Artificial Intelligence Review, 52:2019-2037, 2017.  
Milan Merkle. Inequalities for the gamma function via convexity. Advances in Inequalities for Special Functions, pages 81-100, 2008.  
L. Meunier, B. J. Delattre, A. Araujo, and A. Allauzen. A dynamical system perspective for lipschitz neural networks. International Conference on Machine Learning, 165, 2022.  
M. Mirza and S. Osindero. Conditional generative adversarial nets. arXiv:1411.1784, 2014.  
K. Muandet, K. Fukumizu, B. Sriperumbudur, and B. Schölkopf. Kernel mean embedding of distributions: A review and beyond. Foundations and Trends in Machine Learning, 10:1-144, 2017.  
E. A. Nadaraya. On estimating regression. Theory of Probability and Its Applications, 1964.  
Bao Nguyen, Binh Nguyen, Hieu Trung Nguyen, and Viet Anh Nguyen. Generative conditional distributions by neural (entropic) optimal transport. Proceedings of the 41st International Conference on Machine Learning, 235:37761-37775, 2024.  
O. H. M. Padilla, J. Sharpnack, Y. Chen, and D. M. Witten. Adaptive nonparametric regression with the k-nearest neighbour fused lasso. Technometrics, 107:293-310, 2020.  
G. Papamakarios, T. Pavlakou, and Iain Murray. Masked autoregressive flow for density estimation. Advances in Neural Information Processing Systems, 30, 2017.

P. Pauli, A. Koch, J. Berberich, P. Kohler, and F. Allgower. Training robust neural networks using lipschitz bounds. IEEE Control Systems Letters, 6:121-126, 2022.  
Gabriel Peyré and Marco Cuturi. Computational Optimal Transport: With Applications to Data Science. Foundations and Trends in Machine Learning, 2019.  
G. Ch. Pflug and A. Pichler. From empirical observations to tree models for stochastic optimization: Convergence properties. SIAM Journal on Optimization, 26:1715-1740, 2016.  
M. Rachdi, A. Laksaci, Z. Kaid, A. Benchiha, and F. A. Al-Awadhi. k-nearest neighbors local linear regression for functional and missing data at random. Statistica Neerlandica, 75, 2021.  
P. Ram and K. Sinha. Revisiting kd-tree for nearest neighbor search. KDD'19: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, page 1378-1388, 2019.  
D. Rudolf and N. Schweizer. Perturbation theory for markov chains via Wasserstein distance. Bernoulli, 24:2610-2639, 2018.  
J. J. Ryu and Y. Kim. Minimax regression via adaptive nearest neighbor. arXiv:2202.02464, pages 1447-1451, 2022.  
D. Scott. Multivariate Density Estimation: Theory, Practices, and Visualization. Wiley, 2015.  
D. Shah and Q. Xie. Q-learning with nearest neighbors. Advances in Neural Information Processing Systems, 31, 2010.  
Z. Shi, Y. Wang, H. Zhang, J. Z. Kolter, and Cho-Jui Hsieh. Efficiently computing local lipschitz constants of neural networks via bound propagation. Advances in Neural Information Processing Systems, 35, 2022.  
J. S. Simonoff. Smoothing Methods in Statistics. Springer, 1996.  
S. Singla, S. Singla, and S. Feizi. Improved deterministic l2 robustness on CIFar-10 and CIFar-100. The Tenth International Conference on Learning Representations, 2022.  
C. J. Stone. Optimal global rates of convergence for nonparametric regression. The Annals of Statistics, 10:1040-1053, 1982.  
A. Trockman and J. Z. Kolter. Orthogonalizing convolutional layers with the cayley transform. International Conference on Learning Representations, 36, 2021.  
Y. Tsuzuki, I. Sato, and M. Sugiyama. Lipschitz-margin training: Scalable certification of perturbation invariance for deep neural networks. Advances in Neural Information Processing Systems, 31, 2018.  
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.  
Cédric Villani. Optimal transport: Old and new, volume 338. Springer Science & Business Media, 2008.

A. Virmaux and K. Scaman. Lipschitz regularity of deep neural networks: analysis and efficient estimation. Advances in Neural Information Processing Systems, 31, 2018.  
M. Šmíd and V. Kozmík. Approximation of multistage stochastic programming problems by smoothed quantization. Review of Managerial Science, 2024.  
M. Vuletic, F. Prenzel, and M. Cucuringu. Fin-gan: forecasting and classifying financial time series via generative adversarial networks. Quantitative Finance, 24, 2024.  
M. J. Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge university press, 2019.  
R. Wang and I. Manchester. Direct parameterization of lipschitz-bounded deep networks. Proceedings of the 40th International Conference on Machine Learning, 202, 2023.  
L. Wasserman. All of Nonparametric Statistics. Springer, 2006.  
G. S. Watson. Smooth regression analysis. Sankhya: The Indian Journal of Statistics, Series A, 26: 359-372, 1964.  
T. Xu and B. Acciaio. Conditional cot-gan for video prediction with kernel smoothing. NeurIPS 2022 Workshop on Robustness in Sequence Modeling, 2022.  
A. Xue, L. Lindemann, A. Robey, H. Hassani, G. J. Pappas, and R. Alur. Chordal sparsity for lipschitz constant estimation of deep neural networks. 2022 IEEE 61st Conference on Decision and Control, 6:3389-3396, 2022.  
B. Zhang, D. Jiang, D. He, and L. Wang. Rethinking lipschitz neural networks and certified robustness: A boolean function perspective. Advances in Neural Information Processing Systems, 35, 2022.  
P. Zhao and L. Lai. Minimax regression via adaptive nearest neighbor. 2019 IEEE International Symposium on Information Theory, pages 1447-1451, 2019.  
W. Zhao and E. G. Tabak. Adaptive kernel conditional density estimation. 2023.