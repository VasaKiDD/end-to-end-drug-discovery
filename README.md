# end-to-end-drug-discovery

I am applying a combination of Fractal AI, Genetic Algorithm and Deep Learning for small molecule generation on the benchmark  GuacaMol of Benevolent AI.

* GuacaMol : https://benevolent.ai/guacamol
* Fractal AI : https://arxiv.org/abs/1803.05049
* Graph based genetic algorithm : https://pubs.rsc.org/en/content/articlelanding/2019/SC/C8SC05372C#!divAbstract

I train de Deep Learning discriminator model on Moses Dataseset (SMILES) then I used it as a fitness function in genetic algorithm with entropy maximisation to generate new molecules. Then like adversarial training, I train the discriminator to make the difference between generated molecules and dataset molecules. Finally, to run GuacaMol benchmark, I use a combiunation of deep likeliness with goal oriented task to explore more efficiently the space.

Train discriminator:
```bash
python deep_likeliness/train_discriminator.py
```

Run Gucamol benchmark:
```bash
python guacamol/run_benchmark.py
```
