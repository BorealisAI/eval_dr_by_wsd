# Evaluating quality of dimensionality reduction map with Wasserstein distances

## Code for "Dimensionality Reduction has Quantifiable Imperfections: Two Geometric Bounds" (NeurIPS 2018)

This repo contains the code for calculating the empirical Wasserstein many-to-one and discontinuity.

`example_swissroll_pca.ipynb` illustrates how the calculation is done.

Paper: https://papers.nips.cc/paper/8065-dimensionality-reduction-has-quantifiable-imperfections-two-geometric-bounds

Blog post: https://www.borealisai.com/en/blog/dimensionality-reduction-finally-has-quantifiable-imperfections/

bibtex entry:
```bibtex
@inproceedings{lui2018dimensionality,
  title={Dimensionality Reduction has Quantifiable Imperfections: Two Geometric Bounds},
  author={Lui, Kry and Ding, Gavin Weiguang and Huang, Ruitong and McCann, Robert},
  booktitle={Advances in Neural Information Processing Systems},
  pages={8462--8472},
  year={2018}
}
```

## Dependencies
* [numpy](http://www.numpy.org/)
* [scipy](https://scipy.org/scipylib/)
* [pot](https://pot.readthedocs.io/en/stable/)
* [faiss](https://github.com/facebookresearch/faiss)
* [scikit-learn](https://scikit-learn.org/stable/) (for running the examples)
