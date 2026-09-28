# LLM-BabyBench

This repository contains the code for the paper [**LLM-BabyBench: Can Language Models Plan in Worlds They Can Simulate?**](https://arxiv.org/abs/2505.12135)

Idriss Malek, Omar Choukrani, Daniil Orel, Anh Duy Le Dinh, Zhuohan Xie, Zangir Iklassov, Martin Takáč, Salem Lahlou

## Citation

If you use this work, please cite:

```bibtex
@misc{malek2026llmbabybenchlanguagemodelsplan,
      title={LLM-BabyBench: Can Language Models Plan in Worlds They Can Simulate?},
      author={Idriss Malek and Omar Choukrani and Daniil Orel and Anh Duy Le Dinh and Zhuohan Xie and Zangir Iklassov and Martin Takáč and Salem Lahlou},
      year={2026},
      eprint={2505.12135},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.12135},
}
```

## Datasets
The `Decompose`, `Plan`, and `Predict` datasets are available in the `datasets` folder, along with a clear description.

This dataset is generated using the scripts in the `generate` folder.

## Create and Activate Virtual Environment

To begin, create a Python virtual environment and activate it.

1. **Create the virtual environment:**

```
python -m venv babybench
```

2. **Activate the environment:**

```
source babybench/bin/activate
```

3. **Install the requirements:**

```
pip install -r requirements.txt
```

## Install Minigrid

Next, clone the Minigrid repository and install the necessary dependencies.

1. **Clone the Minigrid repository:**

```
git clone https://github.com/Farama-Foundation/Minigrid.git
```

2. **Checkout the specific commit:**

After cloning the repository, navigate into the project directory and checkout the commit hash:

```
cd Minigrid
git checkout 6e713afef8d23d5280ebf28fb3fcf635d40d6a7f
```

This ensures that you are using the exact version of this repository.

3. **Install Minigrid in editable mode:**

In the same directory, run:

```
python3 -m pip install -e .
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
