# JACS Manuscript Decision: ja-2026-12808m

**Date:** 17 August 2026  
**Manuscript ID:** ja-2026-12808m  
**Title:** Enhancing molecular dynamics with equivariant machine-learned densities  
**Authors:** Bogojeski, Mihail; Hasyim, Muhammad; Vogt-Maranto, Leslie; Müller, Klaus-Robert; Burke, Kieron; Tuckerman, Mark  
**Editor:** Prof. Connor Coley, Associate Editor, *Journal of the American Chemical Society*  
**Decision:** Reject (resubmission invited by 13 February 2027)

---

## Editor letter

17-Aug-2026

Manuscript ID: ja-2026-12808m  
Title: "Enhancing molecular dynamics with equivariant machine-learned densities"  
Author(s): Bogojeski, Mihail; Hasyim, Muhammad; Vogt-Maranto, Leslie; Müller, Klaus-Robert; Burke, Kieron; Tuckerman, Mark

Dear Dr. Tuckerman:

Thank you for submitting your manuscript for publication in the Journal of the American Chemical Society. It has been examined by expert reviewers whose comments are enclosed. The reviewers have expressed serious reservations about this work that I don’t believe could be addressed through a standard major revision. In light of the comments received, I am unable to accept this manuscript for publication in the Journal of the American Chemical Society.

However, if further experimentation, analysis, and revisions allow you to address the reviewers' concerns in full, I am willing to consider a resubmission of this manuscript, if it can be submitted by 13-Feb-2027, at the latest.

Your manuscript is likely to go back to the present reviewers and possibly to one or more additional new reviewers for further input and advice before any final decision on possible publication is made. To help me expedite processing, explicitly address the questions raised by the Editor and reviewers and clearly point out the changes made in the manuscript. Provide this information by attaching a file containing your detailed responses, including explanations for disagreeing with any of the suggestions that you have chosen not to follow. Please annotate a copy of the manuscript to show revisions and track changes for the benefit of the reviewers. This marked manuscript should be uploaded electronically in the File Upload section as "Supporting Information for Review Only".

If you decide not to revise and resubmit for further JACS consideration, please notify me by email (coley-office@jacs.acs.org) at your earliest convenience. Include the following information in the subject line of your email message:

Tuckerman ja-2026-12808m

Sincerely,

Prof. Connor Coley  
Associate Editor  
Journal of the American Chemical Society  
Email: coley-office@jacs.acs.org

---

## Reviewer 1

**Recommendation:** Publish in JACS after minor revisions.

### Comments

Tuckerman and co-workers present in this manuscript DenSNet, a novel machine learning framework that learns electron density distributions from nuclear coordinates. The learned equivariant representations are further used as input to an additional neural network to predict energies (and thus forces) for MD simulations. This setup can be viewed as a transfer learning scheme in which the highly informative electron density, readily accessible from DFT calculations, is first learned by a neural network as a useful target by itself, and the resulting representations are then used to predict energies and forces, which are the usual targets of machine learning FFs. This strategy is expected to significantly improve the robustness and transferability of the resulting MLFF model. The density prediction network itself is also quite useful, in particular in terms of computing spectroscopies. For this part, the authors also introduce several carefully crafted technical details, leading to strong performance. Overall, I regard this as an excellent contribution to the emerging field of machine learning FFs and AI-assisted MD simulations, and shoud thus be accepted for publication. The following comments might further improve the manuscript.

1. A main concern is that the systems tested in the manuscript are still relatively simple. The size extrapolation of polythiophene is convincing, but I wonder how well the method works when non-covalent interactions (weak or strong) play an important role. I would be particularly interested in the performance of DenSNet for hydrogen-bonding systems. A natural test system would be water clusters with different numbers of water molecules, where the model is trained on smaller clusters and tested on larger clusters not seen during training. Simpler systems such as N-Methylacetamide (NMA) dimer or methanol–water dimers, could also be considered. Another set of interesting test systems would be malonaldehyde or AcAc, which contain strong intramolecular hydrogen bonds (proton transfer motifs).

2. Beyond size extrapolation, one of the most important issues for any machine learning FF is its ability to handle out-of-distribution conformations. I feel that this aspect has not been sufficiently explored in the current manuscript. In particular, given the way the training data were constructed, the 1000 selected frames may be highly similar to each other, leading to significant overlap between the training and the validation and test sets. It would be very beneficial for the authors to intentionally generate out-of-distribution conformations for selected systems and evaluate the performance of DenSNet on these structures. Even if the model does not perform well in this setting, such an analysis would still be valuable because it would clarify the limitations and practical domain of applicability of the method.

3. I find the polythiophene case very inspiring, but the current discussion of this part appears disproportionately long. In particular, Figure 6 seems to contain relatively limited information and may be more appropriate for the SI. The comparison of the IR spectra here is mainly against DFT reference calculations rather than direct experimental measurements. If the authors prefer to keep some of the Figure 6 results in the main text, one possible reorganization would be to combine Figure 5a with Figure 4, where it may fit more naturally, and then combine selected Figure 6 results, either as a figure or a table, with Figure 5b. This would make the main text more focused.

### Minor points

4. Figure 7 appears to be relatively standard and may be better placed in the SI rather than the main text.

5. The authors should provide a more detailed description of the datasets (SI S2.2). For example, for the original datasets, how much overlap exists between the training and test geometries? For resorcinol and ethanethiol, the authors only mention classical MD simulations, but important details are missing. How long were these simulations? What FF model was used? Were hydrogen atoms constrained or not? These details should be provided. "The test samples were randomly selected from a separate set of classical MD trajectories at 300 K, ensuring no overlap with the training points." does not seem a valid claim to me. Running an additional classical MD simulation does not by itself guarantee that the training and test conformations are non-overlapping. The authors should provide evidence supporting this claim, for example by quantifying structural or descriptor-space differences between the training and test sets.

6. Some parts of the writing, especially in the later sections, should be tightened. For example, the timing result (755 ms vs 5 min) appears twice in the manuscript. This repetition seems to be a mistake made during the manuscript revision.

7. In the Discussion section, several speculative statements should be removed or more carefully qualified. For example, "For solvated molecules and hydrogen- bond networks, DenSNet also enables computation of dipole autocorrelation functions and frequency-dependent dielectric response from large-scale trajectories." (page 25). I don't think this statement is currently supported, although I would really like to see this tested, as reflected in my point 1. The cited reference also does not seem directly relevant. The authors should carefully examine and revise the discussion to ensure that all claims are supported by the actual results.

### Additional questions

| Question | Answer |
| --- | --- |
| Significance | High (suitable for JACS) |
| Novelty | Highest (top 5%, suitable for JACS) |
| Broad interest | High (suitable for JACS) |
| Scholarly presentation | High (suitable for JACS) |
| Are the conclusions adequately supported by the data? | Yes |
| Are the literature references appropriate and correct? | Yes |
| Do the contents justify the length of this manuscript? | Yes |

---

## Reviewer 2

**Recommendation:** Publish elsewhere.

### Comments

The work by Bogojeski et al. introduces DenSNet, an equivariant network based on a repurposed PhiSNet to predict electron densities and a companion network deriving energies and forces. In tandem, they are used to drive AIMD-quality IR spectra simulations as demonstrated for three small molecules and thiophene oligomers. Technically, it is an interesting work and manuscript is well written, although impact and novelty are rather limited and, hence, I would recommend resubmitting this work to JPCA or another specialized journal.

1. Conceptually, there are works with a similar idea of predicting electron densities, deriving from there energies and forces, and ultimately IR spectra from AIMD, e.g., Hazra, S.; Patil, U.; Sanvito, S. A charge-density machine-learning workflow for computing the infrared spectrum of molecules. *J. Chem. Phys.* 2025, 163 (17). DOI: [10.1063/5.0292221](https://doi.org/10.1063/5.0292221).

2. Practically, there are more general and faster approaches to calculate IR spectra, which work out-of-the-box across chemical space; some of which the authors cited, e.g., MACE4IRmol and AIMNet-2, but also other MACE-based variants such as [10.1021/acs.jctc.4c01157](https://doi.org/10.1021/acs.jctc.4c01157) and conceptually different [TranSpec](https://doi.org/10.1021/jacs.5c05010) and [AIQM](https://doi.org/10.26434/chemrxiv-2024-604wb). Now the authors' approach needs to be fitted for every new system and will be slower anyways.

3. Extrapolation to larger systems is an important experiment but it does not show the benefit of using this architecture nor does it demonstrate that it can handle really-tough extrapolation cases which are a challenge to machine learning potentials. The training data contains all oligomers up to hexamer, at which lengths the spectra already converge to the polymer, and even then, error increase trastically for longer oligomers.

4. From the main text, it is unclear how the companion network works exactly. Does it kind of 'embed' density network? Is the autodifferentiation then also takes derivative of density? Related: is the AIMD energy conserving?

5. It seems that companion network only produces energies and forces, but main text (Page 4) implies that it also produces 'electronic observables'. How is the dipole moment calculated, directly from the density network or is the companion network involved?

6. Page 5 talks about a single equivariant machine-learning framework, but there are two networks. Are they both used in the same time during MD propagation without any post-processing step?

7. It is unclear how normalization by the total electron count ensures the errors in valence and core regions weighted by their contributions. At the end, the authors used delta-learning approach which resolves this problem properly - but if normalization worked then they wouldn't need delta-learning?

8. Since delta-learning should correct for excessive density, i.e., delta-density should be negative in some regions. How does that reconcile with enforcing nonnegativity with softplus transformation?

9. The maximum propagation time in AIMD should be mentioned in the main text.

10. Units for density errors are not given.

11. There is no numerical evidence given that DenSNet provides equilibrium structures similar to DFT, despite the claim.

12. One more relevant reference missing in the introduction is Cuevas-Zuviria, B.; Pacios, L. F. Analytical Model of Electron Density and Its Machine Learning Inference. *J. Chem. Inf. Model.* 2020, 60 (8), 3831–3842. DOI: [10.1021/acs.jcim.0c00197](https://doi.org/10.1021/acs.jcim.0c00197).

13. It is unclear who are the corresponding authors, main manuscript and SI files disagree.

### Additional questions

| Question | Answer |
| --- | --- |
| Significance | Moderate (not suitable for JACS) |
| Novelty | Moderate (not suitable for JACS) |
| Broad interest | Moderate (not suitable for JACS) |
| Scholarly presentation | High (suitable for JACS) |
| Are the conclusions adequately supported by the data? | Yes |
| Are the literature references appropriate and correct? | In Part |
| Do the contents justify the length of this manuscript? | Yes |

---

## Reviewer 3

**Recommendation:** Reconsider after major revisions.

### Comments

The authors introduce DenSNet, a novel ML framework that learns the electron density directly from nuclear coordinates and predicts both electronic properties and atomic forces. The work is timely and relevant because most existing ML approaches focus primarily on predicting energies, forces, and, in some cases, atomic charges. In contrast, the simultaneous prediction of electron densities together with electronic observables, such as dipole moments and IR spectra, within a single framework remains relatively unexplored.

DenSNet can be viewed as an extension of the previously published PhiSNet architecture. Although it builds upon existing ideas, the authors introduce several significant modifications that substantially improve the original methodology. Overall, DenSNet demonstrates promising performance compared with the baseline density-fitting (DF) approach used throughout the manuscript. Nevertheless, I have several major and minor concerns regarding the general applicability, benchmarking, and validation of the proposed framework. I encourage the authors to address the following comments before the manuscript can be considered for publication in JACS.

### Major comments

1. I was surprised that all benchmark datasets were generated using the PBE exchange-correlation functional. It is well established that PBE does not always provide an accurate description of electronic interactions in molecular systems, particularly for properties that are highly sensitive to the electron density. Hybrid density functionals generally provide significantly improved accuracy. Is there any fundamental limitation of the DenSNet framework that prevents it from learning electron densities computed at the hybrid DFT level? From a quantum chemistry perspective, it would be highly valuable to evaluate the model on electron densities computed with more accurate electronic structure methods, such as PBE0+MBD or ωB97M-V/def2-TZVPD. Such an analysis would help clarify whether the framework itself is sufficiently general or whether its current performance is primarily limited by the underlying reference data.

2. This point is closely related to the previous comment. The manuscript lacks comparisons with current state-of-the-art ML force fields. To assess the broader applicability of DenSNet, comparisons with general-purpose ML potentials would be highly informative, particularly for IR spectrum calculations. Several well-established benchmarks are already available, including Crambin in water (*J. Am. Chem. Soc.* 2025, 147, 33723–33734), paracetamol (*J. Am. Chem. Soc.* 2025, 147, 17598–17611), and phenylalanyl-glycyl-glycine (*Chem. Eur. J.* 2005, 11, 6803–6817). Comparing DenSNet against models such as SO3LR and MACE-OFF would clearly demonstrate the advantages (or limitations) of explicitly learning the electron density rather than relying on conventional ML force fields. It would also be valuable to include comparisons with semi-empirical approaches, such as GFN-xTB and DFTB. In addition, Figure 2 should be expanded to include these methods in the computational cost analysis. Such comparisons would better establish the practical relevance of the proposed methodology and should be discussed in the main text.

3. Another concern relates to the interpretation of the reported electron density errors. Although the reported mean absolute errors are very small, it would be difficult for the readers to assess whether these values correspond to chemically meaningful accuracy. For energies and forces, well-established thresholds (e.g., approximately 1 kcal/mol and 1 kcal/mol/Å) provide useful reference points. No analogous guideline exists for electron density prediction. Reporting relative errors, in addition to absolute errors, would provide a more intuitive assessment of model accuracy. Furthermore, I encourage the authors to quantify the uncertainty of their predictions, which would help the community better interpret the reported performance and facilitate future comparisons.

4. The delta learning approach represents one of the major contributions of this work. Previous studies have demonstrated that delta learning often improves predictive accuracy and transferability compared with direct learning. However, this increased accuracy comes at the expense of computing the baseline quantities, which in this work correspond to the SAD prior. Therefore, the manuscript should compare the computational cost of the direct and delta learning approaches. Such an analysis would help determine whether the proposed strategy remains computationally advantageous for larger and more complex systems.

   Furthermore, for energy and force prediction it is well known that delta learning naturally produces lower MAEs because the correction term has a much smaller magnitude than the original target. Whether the same argument applies to electron density prediction remains unclear. Since this is among the first studies proposing delta learning for electron density corrections, it would be valuable to characterize the magnitude of these corrections and compare direct and delta learning models using relative errors rather than only absolute errors.

5. The manuscript provides little information regarding the hyperparameter optimization used to determine the final model architecture. Instead, the architectural parameters appear to have been selected without a clearly defined optimization procedure. This analysis is particularly important because the reported performance differences between the direct and delta learning models are relatively small. Alternative hyperparameter choices could potentially reduce these differences or even alter which model performs best. Based on previous studies, parameters such as the cutoff radius are especially important because they strongly influence model accuracy, scalability, and transferability.

6. For the IR spectra of the small molecules, the authors state that each trajectory was propagated for a "sufficient duration" to obtain converged spectra. This statement requires further clarification. What trajectory lengths were actually used (e.g., 1 ns, 5 ns, or 10 ns)? How was convergence quantified? Was the spectrum monitored as a function of simulation length?

   A similar concern applies to the polythiophene oligomers, where only 100 ps ML-MD simulations were performed. Was this simulation length selected based on a convergence analysis? How would the predicted spectra change if the simulations were extended to 200 ps or longer? One of the principal advantages of ML force fields is the ability to perform substantially longer molecular dynamics simulations than conventional electronic structure methods. Therefore, I encourage the authors to provide stronger evidence that the selected simulation lengths are sufficient for obtaining converged IR spectra.

7. It would be important to clarify whether the proposed methodology can be extended to develop a general-purpose model capable of predicting the electron density across a broad range of molecular systems, or whether such an approach is currently impractical due to the computational cost associated with generating the reference data. Moreover, what strategy would be most appropriate for constructing a comprehensive training dataset for such a model? For example, could active learning be employed to efficiently sample the chemical space while minimizing the number of expensive quantum mechanical calculations?

8. Finally, I recommend moderating some of the claims regarding the agreement between the predicted and reference spectra. Throughout the manuscript, the authors repeatedly state that the calculated spectra are in close agreement with, or match, the experimental or computational reference data across the entire frequency range. While the peak positions generally agree well, noticeable discrepancies remain in the peak intensities and relative spectral areas. Unless additional validation is provided, these statements should be softened to more accurately reflect the presented results.

### Minor comments

1. Please provide additional computational details regarding the two-stage training procedure, including the computational cost of training both the direct and delta learning models, particularly for the larger systems.

2. Figure 1 could be improved by incorporating the model architecture directly into the workflow. This would provide readers with a more complete overview of the methodology at the beginning of the manuscript.

3. The manuscript would benefit from a broader discussion of which additional electronic properties could potentially be computed from the predicted electron density using DenSNet. Such a discussion would help readers better appreciate the future scope and versatility of the proposed framework.

### Additional questions

| Question | Answer |
| --- | --- |
| Significance | Highest (top 5%, suitable for JACS) |
| Novelty | High (suitable for JACS) |
| Broad interest | High (suitable for JACS) |
| Scholarly presentation | High (suitable for JACS) |
| Are the conclusions adequately supported by the data? | In Part |
| Are the literature references appropriate and correct? | Yes |
| Do the contents justify the length of this manuscript? | In Part |

---

## Publisher notes

If the Editor has suggested that your manuscript may benefit from language editing, you are encouraged to have it reviewed by a fluent English speaker before resubmitting. ACS Authoring Services can assist you by matching your manuscript with trained chemists who will edit your work for accurate terminology and phrasing. For pricing and information, visit [authoringservices.acs.org](https://authoringservices.acs.org/). Please note that using this (or any) service does not guarantee that your manuscript will be accepted for publication.

**For assistance with manuscript submission, contact:**

ACS Publications Customer Services & Information (CSI)  
Email: support@services.acs.org  
Phone: 202-872-4357  
Toll Free Phone: 800-227-9919 (USA/Canada only)
