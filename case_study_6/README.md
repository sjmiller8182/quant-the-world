# Searching for the Higgs Boson with Deep Learning

## Summary

A Higgs boson is an elementary particle produced by quantum application of energy within the Higgs field, which can be used to explain why particles have mass. Because the process responsible for creating bosons was previously unknown to the researchers, the researchers generated sample process signals using Monte Carlo simulations to help build a model that can classify if a signal process is likely to produce a boson or if the signal process is only likely to produce noise. Monte Carlo simulations are useful for studying the property of tests when the assumptions they are derived from are not met. For this reason, the researchers applied this method. Because deep learning is useful for gaining inference from massive amounts of data when there are only very minor differences responsible for class separation and the Monte Carlo simulations generated 11 million signals, the researchers decided to model the signals using a deep neural network. Altogether, the research performs classification on 28 total features; 21 are kinetic properties and an additional seven are derived from functions of those properties, which are used to discriminate between the two classes. The two classes are signals processes that create Higgs bosons and signal processes that do not. In this project, we deconstruct the research and reconstruct the modeling performed. We then analyze the methods used and determine if a more useful approach exists given the enhancement of technology available since the original research concluded in 2014.

[See Full Report](./Searching%20for%20the%20Higgs%20Boson%20with%20Deep%20Learning.pdf)

**Language**: python

## Data

The HIGGS Data Set was used for this analysis. See [https://archive.ics.uci.edu/ml/datasets/HIGGS](https://archive.ics.uci.edu/ml/datasets/HIGGS) for more details.

## References

* \[1\] Baldi, P., P. Sadowski, and D. Whiteson. “Searching for Exotic Particles in High-energy Physics with Deep Learning.” Nature Communications 5 (July 2, 2014). [https://arxiv.org/pdf/1402.4735.pdf](https://arxiv.org/pdf/1402.4735.pdf)

