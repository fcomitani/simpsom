---
title: 'SimpSOM : Self-Organizing Maps made simple'
tags:
  - Python
  - self-organizing maps
  - neural network
  - dimensionality reduction
  - clustering
authors:
  - name: Federico Comitani
    orcid: 0000-0003-3226-1179
    affiliation: "1, 2" 
    corresponding: true
  - name: Simone Riva
    affiliation: "3" 
  - name: Andrea Tangherloni
    affiliation: "3" 
affiliations:
  - name: Program in Genetics and Genome Biology, The Hospital for Sick Children, Toronto, Ontario, Canada
    index: 1
  - name: Cyclica Inc., Toronto, Ontario, Canada
    index: 2
  - name: -???
    index: 3
date: 1 December 2022
bibliography: paper.bib

# Summary

Simple Self-Organizing Maps (SimpSOM) is a lightweight implementation of Kohonen self-organizing maps natively implemented in Python. 

<!-- what are SOM -->


is a general-purpose algorithm 
for approximating the Pareto set of multi-objective optimization problems [@moead]. It decomposes the original 
multi-objective problem into a number of 
single-objective optimization sub-problems and then uses an evolutionary process to optimize these 
sub-problems simultaneously and cooperatively. MOEA/D is a state-of-the-art algorithm in aggregation-based 
approaches for multi-objective optimization.

The goal of the *moead-framework* python package is to provide a modular framework for scientists and 
researchers interested in experimenting with MOEA/D and its numerous variants.

<!-- new version -->

# Statement of Need

<!-- current state of the art -->

SOMPY matlab implementation of somtoolbox only batch and only square [@moosavi2014]
minisom no pbc[@vettigli2018]
neupy is tensorflow based (no citation, https://github.com/itdxer/neupy)
som is basically a copy of minisom (no citation, https://github.com/alexarnimueller/som), only square but has PBC
kohonen is basic (no citation, https://github.com/lmjohns3/kohonen), different from others
GEMA only square no pbc [@garciatejodor2021]

<!-- features -->


OBJ oriented + online, batch + hexagonal, square + gassuian, bubble, mexican hat + PBC + PCA
any sklearn-compatible metric,  sklearn-integrated
GPU with CUPY and CUML, optional
sklearn-compatible clustering (flexible)
plotting


<!-- used by -->

> Postema, J. T. (2019). Explaining system behaviour in radar systems (Master's thesis, University of Twente).
[@postema2019]
> Lorenzi, C., Barriere, S., Villemin, J. P., Dejardin Bretones, L., Mancheron, A., & Ritchie, W. (2020). iMOKA: k-mer based software to analyze large collections of sequencing data. Genome biology, 21(1), 1-19.
[@lorenzi2020]
> Saunders, J. K., McIlvin, M. R., Dupont, C. L., Kaul, D., Moran, D. M., Horner, T., ... & Saito, M. A. (2022). Microbial functional diversity across biogeochemical provinces in the central Pacific Ocean. Proceedings of the National Academy of Sciences, 119(37), e2200014119.
[@saunders2022]





The MOEA/D algorithm is now considered as a framework. MOEA/D is the basis of many variants that improve or 
add new components to improve MOEA/D performance.
The first version of MOEA/D and its most famous variants [@moead_de; @moead_dra] are implemented in recent multi-objective 
optimization software such as pymoo [@pymoo], pygmo [@pygmo] and jMetal [@jmetal]. These implementations offer 
many state-of-the-art algorithms, visualization tools or parallelization abstraction, but they do not enable detailed 
testing and analysis of the various algorithm's components' behavior.
The modular R package MOEADr [@Campelo_2020] focuses on MOEA/D and allows the definition of different variants for 
each component of MOEA/D. While some modular frameworks already exist in Python for evolutionary algorithms 
such as DEAP [@DEAP_JMLR2012] or ModEA [@vanrijn2016], these do not (easily) support implementing MOEA/D variants. 
Instead, they focus mostly on single-objective optimization and CMA-ES variants respectively.

With the *moead-framework* package, we aim to provide the modularity of the MOEADr package by 
using the flexibility of Python. Indeed, we want to allow the user to update the behavior of MOEA/D 
components in their research works without being limited by the software. 
The package is focused on a modular architecture for easily adding, updating or testing the components of 
MOEA/D and for customizing how components interact with each other. Indeed, in contrast with other existing implementations, 
*moead-framework* does not limit the users with a limited number of components available as parameters (8 components are available 
in MOEADr). Users can easily restructure the 10 existing components of the *moead-framework* and include new ones to easily add new features without 
altering existing components. Components are not only customizable with parameters as with MOEADr, but in fact they can be added
with the inheritance mechanism on the main run() method of each algorithm.

For example, the *moead-framework* package was used for creating novel sub-problem selection strategies and 
analyzing them [@gpruvost_evocop2020], and for rewriting the component used to generate 
new candidate (offspring) solutions with a variant based on Walsh surrogates [@gpruvost_gecco2020].




# How to Use


<!-- brief MNIST example -->
![Pipeline for neural predictions for syntax guided program synthesis.\label{fig:description}](sygus.png)


# Documentation

<!-- documentation -->
The documentation is available at the following URL: 
[moead-framework.github.io/framework/](https://moead-framework.github.io/framework/html/index.html).

A [complete example](https://moead-framework.github.io/framework/html/examples.html) and 
[all components](https://moead-framework.github.io/framework/html/api.html) are described in details.
[Two tutorials](https://moead-framework.github.io/framework/html/tuto.html) are made available for the user 
to experiment with their own multi-objective optimization problem and to implement their own MOEA/D variants.


# Acknowledgements



# References
