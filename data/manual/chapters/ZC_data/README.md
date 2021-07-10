# Semantic Scholar Analysis

These files were generated using the script `code/chapters/ZC_data/search_semantic_scholar.py`.

To re-generate this data, download the Semantic Scholar Open Corpus from April 2021 and run
the following commands in the folder containing the downloaded data:

```sh
search_semantic_scholar.py --cites 9664ab17170e0442f35e27d8b3fac7398aa12d08 6f093c5ec100804c98a25d9831b025949e5cb78e 939be55e4c179b8f306445e85656d437282ec1d9 | tee semantic_scholar_open_corpus_2021_04_nef_citations.json

search_semantic_scholar.py "Nengo" "Neural Engineering Framework" | tee semantic_scholar_open_corpus_2021_04_nef_nengo.json
```

Compress the data using `xz`:
```sh
xz semantic_scholar_open_corpus_2021_04_nef_citations.json
xz semantic_scholar_open_corpus_2021_04_nef_nengo.json
```

# Two-Compartment LIF Optimal Regularisation and Filters

The file `two_comp_benchmark_optimal_points.json` was created by manually concatenating the output of the plotting scripts
`two_comp_benchmark_functions_regularisation_filter_sweep_nosubth.py` and `two_comp_benchmark_functions_regularisation_filter_sweep_subth.py`.
