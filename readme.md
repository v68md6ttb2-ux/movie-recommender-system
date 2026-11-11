# MovieLens Recommender System: Scikit-learn - PyTorch

This project implements collaborative and content-based movie recommendation models using the [MovieLens](https://grouplens.org/datasets/movielens/) dataset. 

It includes two approaches:

- **Scikit-learn**: Collaborative filtering (matrix factorization) and content-based filtering (TF-IDF).
- **PyTorch**: Neural Collaborative Filtering (NCF) with embeddings.

Both models expose simple APIs for retrieving recommendations, and the project includes comprehensive analysis and visualization tools for model comparison.


## Getting Started

### 1. Download the Dataset

Run the following script to download and extract the MovieLens dataset:

```sh
python download_data.py
```

### 2. Install Dependencies

Install required Python packages:

```sh
pip install -r requirements.txt
```

### 3. Training & Evaluation

#### * PyTorch Model

- See [`pytorch/recommender_pytorch.ipynb`](pytorch/recommender_pytorch.ipynb) for a notebook implementation.
- Run [`pytorch/main.py`](pytorch/main.py) to start a FastAPI server to view recommendations:
```sh
cd pytorch
uvicorn main:app --reload
```

Artifacts:
- `models/neural_cf_model.pth`: Trained PyTorch model weights
- `models/user_id_to_index.json`, `models/movie_id_to_index.json`: ID mappings

#### * Scikit-learn Model

- See [`scikit_learn/recommender_scikit.ipynb`](scikit_learn/recommender_scikit.ipynb) for collaborative and content-based filtering.

A Flask API is provided in [`scikit_learn/app.py`](scikit_learn/app.py):
```sh
python scikit_learn/app.py
```

Endpoints:
- `/recommend/collab/<user_id>`: Collaborative filtering recommendations
- `/recommend/content/<movie_id>`: Content-based recommendations

Artifacts:
- `models/recommender_scikit_artifacts.pkl`: Scikit-learn model artifacts

### 4. Model Analysis and Visualization

The project includes comprehensive analysis tools to compare models and visualize results:

#### Run Model Analysis

```sh
python analyze_models.py
```

This script generates:
- Performance comparison charts (`performance_comparison.png`)
- Training curve visualization (`training_curve.png`)
- Ablation study charts:
  - Embedding dimension analysis (`ablation_embedding.png`)
  - Network layer analysis (`ablation_layers.png`)

#### Collect All Results

```sh
python collect_all_results.py
```

This script compiles all model results, performance metrics, and analysis into a single report file (`all_models_results.txt`).

## Project Structure

```
ml_project/
├── analyze_models.py           # Model comparison and visualization
├── collect_all_results.py      # Results aggregation script
├── data/                       # Dataset directory
│   └── ml-latest-small/        # MovieLens dataset
├── download_data.py            # Dataset download script
├── pytorch/                    # PyTorch implementation
│   ├── main.py                 # FastAPI server
│   ├── models/                 # PyTorch model artifacts
│   └── recommender_pytorch.ipynb  # PyTorch implementation notebook
├── readme.md                   # This file
├── requirements.txt            # Dependencies
├── scikit_learn/               # Scikit-learn implementation
│   ├── app.py                  # Flask server
│   ├── models/                 # Scikit-learn model artifacts
│   └── recommender_scikit.ipynb    # Scikit-learn implementation notebook
├── test_pytorch_model.py       # PyTorch model testing
└── test_scikit_model.py        # Scikit-learn model testing
```

## Requirements

See [`requirements.txt`](requirements.txt) for dependencies. Main packages include:
- pandas
- numpy
- scikit-learn
- scipy
- torch
- torchvision
- flask
- fastapi
- uvicorn
- joblib
- matplotlib
- seaborn

## Data

The MovieLens dataset is stored in `data/ml-latest-small/`. See [`data/ml-latest-small/README.txt`](data/ml-latest-small/README.txt) for details.

## Model Comparison

| Model | RMSE | Precision@10 | Training Time |
|-------|------|-------------|---------------|
| Scikit-learn (SVD) | 0.9066 | 0.1016 | Fast |
| PyTorch (NCF) | 0.9643 | 0.0892 | Medium |

### Key Findings

- **Scikit-learn SVD**: Higher accuracy, faster training, recommended mainstream movies
- **PyTorch NCF**: Better diversity, more discoverability, recommended niche movies

## License

See the MovieLens dataset license in [`data/ml-latest-small/README.txt`](data/ml-latest-small/README.txt).

## References

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. [https://doi.org/10.1145/2827872](https://doi.org/10.1145/2827872)
- Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua. 2017. Neural Collaborative Filtering. Proceedings of the 26th International Conference on World Wide Web (WWW '17). ACM, Perth, Australia, 173-182. [https://doi.org/10.1145/3038912.3052569](https://doi.org/10.1145/3038912.3052569)