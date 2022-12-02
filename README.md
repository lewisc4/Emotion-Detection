# Emotion-Detection Using Deep Learning
Fine-tuning a residual neural network to classify human emotions based on images of human faces.

## Project Overview
In this project, a pre-trained [ResNet model](https://arxiv.org/pdf/1512.03385.pdf), specifically [this implementation](https://huggingface.co/microsoft/resnet-18), is fine-tuned on the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) to classify 7 human emotions. It is shown that this fine-tuned model achieves higher accuracies than several baseline models as well as a higher accuracy than human evaluations. The training results and metrics for this project were primarily tracked using [wandb](https://wandb.ai/site) and the associated training runs can be found [here](https://wandb.ai/clewis7744/emotion_detection).

## Setting Up The Environment
### Package Installation
It is necessary to have python >= 3.7 installed in order to run the code for this project. In order to install the necessary libraries and modules follow the below instructions.

1. Clone or download this project to your local computer.
2. Navigate to the `code\` directory, where the `setup.py` file is located.
3. Install the `emotion_detection` module and all dependencies by running the following command from the CLI: `pip install -e .` (required python modules are in `requirements.txt`). **GPU specific instructions:**
    1. If you are on a GPU machine, you need to install a GPU version of pytorch. To do that, first check what CUDA version your server has with `nvidia-smi`.
    2. If your CUDA version is below 10.2, don't use this server
    3. If your CUDA version is below 11, run `pip install torch`
    4. Else, `pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
    5. Check that pytorch-GPU works via `python -c "import torch; print(torch.cuda.is_available())"`. If it returns False, reinstall pytorch via one of the above command (usually this helps).
    6. If you are using 30XX, A100 or A6000 GPU, you have to use CUDA 11.3 and above. 

## Training A Model
### Hyperparameters
The available hyperparameters for fine-tuning the ResNet model can be found in the `emotion_detection/utils.py` file. By default, a large majority of the hyperparameters are inherited from the ResNet model's original parameters. The default model is `microsoft/resnet-18` (shouldn't be changed). However, useful parameters to change/test with are:

* `output_dir` <- Where to save the model to (defaults to `code/cli/outputs/`)
* `learning_rate` <- The external learning rate
* `batch_size` <- Batch size used by the model
* `weight_decay` <- The external weight decay
* `eval_every_steps` <- How often to evaluate the model (compute eval accuracy)
* `debug` <- Whether to run in debug mode (uses small number of examples) or not
* `num_train_epochs` <- Number of training epochs to use

### CLI Commands
**Make sure you have followed the `Setting Up The Environment` section before running these commands**

The below commands can be run from the `cli` directory. By default, the model is saved to the `code/cli/outputs/` directory. If the provided `output_dir` does not exist, it will automatically be created.

**To train a model that perfectly fits a small set of training examples:**

**To train a model with the parameters that achieved the best accuracy:**

**To perform wandb sweeps using the `sweep.yaml` configuration file (make sure you have wandb project)**
