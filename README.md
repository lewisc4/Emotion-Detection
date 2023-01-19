# Facial Expression Recognition (FER) Using Deep Learning

## Project Overview
In this project, a pre-trained [ResNet model](https://arxiv.org/pdf/1512.03385.pdf) (specifically [this implementation](https://huggingface.co/microsoft/resnet-18)), was fine-tuned on the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) to classify 7 human emotions based on images of human faces. It is shown that this fine-tuned model achieves higher accuracies than several baseline models as well as a higher accuracy than human evaluations. The training results and metrics for this project were primarily tracked using [wandb](https://wandb.ai/site) and all associated training runs can be found [here](https://wandb.ai/clewis7744/emotion_detection), with the best training run being [here](https://wandb.ai/clewis7744/emotion_detection/runs/3jct8bsf).

## Setting Up The Environment
### Package Installation
It is necessary to have python >= 3.7 installed in order to run the code for this project. In order to install the necessary libraries and modules follow the below instructions.

1. Clone or download this project to your local computer.
2. Navigate to the `code\` directory, where the `setup.py` file is located.
3. Install the `emotion_detection` module and all dependencies by running the following command from the CLI: `pip install -e .` (required python modules are in `requirements.txt`).
 
### GPU-related Requirements/Installations
1. If you are on a GPU machine, you need to install a GPU version of pytorch. To do that, first check what CUDA version your server has with `nvidia-smi`.
2. If your CUDA version is below 10.2, don't use this server
3. If your CUDA version is below 11, run `pip install torch`
4. Else, `pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
5. Check that pytorch-GPU works via `python -c "import torch; print(torch.cuda.is_available())"`. If it returns False, reinstall pytorch via one of the above command (usually this helps).
6. If you are using 30XX, A100 or A6000 GPU, you have to use CUDA 11.3 and above.


### Downloading The Dataset
The dataset can be downloaded from [here](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz). Specifically, the `fer2013.tar.gz` file should be downloaded and unzipped/extracted. It contains all of the training and testing examples in one `.csv` file under `fer2013/fer2013.csv`.

After the dataset has been downloaded, you can run `python3 make_datasets.py` to extract the training and testing files. By default, the Kaggle dataset should be downloaded and unzipped in the `code/cli` directory. However, you can override the input directory via the script's `--input_data_dir` argument. By default, the output files will be stored in the `cli/dataset/fer2013` folder, which contains the `train.csv`, `public_test.csv`, and `private_test.csv`. However, you can override this via the script's `--output_data_dir` argument. After the `cli/dataset/fer2013` folder and files have been created, the downloaded dataset is no longer needed and the training script can be run (see below).

## Training A Model
### Hyperparameters
The available hyperparameters for fine-tuning the ResNet model can be found in the `emotion_detection/utils.py` file. By default, a large majority of the hyperparameters are inherited from the ResNet model's original parameters. The default model is `microsoft/resnet-18`. Useful parameters to change/test with are:

* `data_dir` <- Parent folder of the dataset (`dataset` by default). See the `Downloading The Dataset` section for more.
* `output_dir` <- Where to save the model to (defaults to `code/cli/outputs/`)
* `test_for_val` <- Whether to use the test set for validation or not. If not, a subset of the training data is used.
* `test_type` <- Uses either public test set (`public_test.csv`) or the private test set (`private_test.csv`) if the test set is used for validation.
* `percent_train` <- What percentage of the training dataset should be used for training, if a subset is used as a validation set.
* `learning_rate` <- The external learning rate
* `batch_size` <- Batch size used by the model
* `weight_decay` <- The external weight decay
* `eval_every_steps` <- How often to evaluate the model (compute eval accuracy)
* `debug` <- Whether to run in debug mode (uses small number of examples) or not
* `num_train_epochs` <- Number of training epochs to use
* `wandb_project` <- The weights and biases project to use (not required)
* `use_wandb` <- Whether to log to weights and biases or not (do not use unless you have a project set via `wandb_project`)

### CLI Training Commands
**Make sure you have followed the `Setting Up The Environment` section before running these commands**

The below commands can be run from the `cli` directory. By default, the model is saved to the `code/cli/outputs/` directory. If the provided `output_dir` does not exist, it will automatically be created.

**To train a model with the parameters that achieved the best accuracy:**

`python3 train.py --test_for_val --test_type='public_test' --pretrained_model_name='microsoft/resnet-50' --batch_size=64 --learning_rate=1e-3 --lr_scheduler_type='linear' --weight_decay=0.0 --num_train_epochs=30 --eval_every_steps=90 --logging_steps=90 --checkpoint_every_steps=10000 --seed=42`

**To perform wandb sweeps using the `sweep.yaml` configuration file (make sure you have set a wandb project using the wandb_project argument):**

1. `wandb sweep --project emotion_detection sweep.yaml`
2. `wandb agent wandb_username/emotion_detection/sweep_id`
