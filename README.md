# casForest
Script for classification with Cascade Deep Forest ([Zhou & Feng, 2019](https://doi.org/10.1093/nsr/nwy108)) combining Random Forest ([Ho, 1995](https://doi.org/10.1109/ICDAR.1995.598994); [Breiman, 2001](https://doi.org/10.1023%2FA%3A1010933404324)) and Extremely Randomized Trees ([Geurts et al, 2006](https://link.springer.com/article/10.1007/S10994-006-6226-1)). The Iris dataset ([Anderson, 1935](https://wiki.irises.org/pub/Hist/Info1986SIGNA37/SIGNA_37.pdf); [Anderson, 1936](https://doi.org/10.2307/2394164); [Fisher, 1936](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x)) is used for demonstration. The implementation is compatible with MATLAB R2019b.

Usage:

```matlab
call_generic_casForest()
```

<p align="center">
    <img src="figure.png" alt="example image" height="500">
</p>

Cite As

[Nzakimuena, C. B. (2020). Automated analysis of retinal and choroidal OCT and OCTA images in AMD. Ecole Polytechnique, Montreal (Canada).](https://publications.polymtl.ca/5234/)

### References

1. [Zhou, Z. H., & Feng, J. (2019). Deep forest. National science review, 6(1), 74-86.](https://doi.org/10.1093/nsr/nwy108)
1. [Ho, T. K. (1995, August). Random decision forests. In Proceedings of 3rd international conference on document analysis and recognition (Vol. 1, pp. 278-282). IEEE.](https://doi.org/10.1109/ICDAR.1995.598994)
1. [Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.](https://doi.org/10.1023%2FA%3A1010933404324)
1. [Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. Machine learning, 63(1), 3-42.](https://link.springer.com/article/10.1007/S10994-006-6226-1)
1. [Anderson, E. (1935). The irises of the Gaspe Peninsula. Bulletin of American Iris Society, 59, 2-5.](https://wiki.irises.org/pub/Hist/Info1986SIGNA37/SIGNA_37.pdf)
1. [Anderson, E. (1936). The species problem in Iris. Annals of the Missouri Botanical Garden, 23(3), 457-509.](https://doi.org/10.2307/2394164)
1. [Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of eugenics, 7(2), 179-188.](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x)
