# Perfectly Accurate Membership Inference by a Dishonest Central Server in Federated Learning

This code can be used to reproduce the findings in 
`Perfectly Accurate Membership Inference by a Dishonest Central Server in Federated Learning`.

## Jupyter Notebook

### Quickstart
 1. Set up a Python environment that satisfies `requirements.txt`.
 2. Start a Jupyter Notebook in `./`.
 3. Navigate to `main.ipynb`.
 4. Change the parameters as necessary and perform the experiment.
 5. The results are saved in `output/`.
 6. The same notebook `main.ipynb` can be used to print the results.
   
The original values are saved in `results.orig.json` and can also be explored.

### Docker container
A `Dockerfile` is included to provide a containarized version of the code, taking care of the creation and setting up of the Python environment with all the required packages. A working Docker environment is needed. There are several tutorials for setting up a docker installation, e.g., [this one](https://docs.docker.com/engine/install/). The following guide has been written for a Linux based system. 
Steps required to run the Jupyter Notebook in a docker container:
1. Navigate to the folder where the Dockerfile is stored (root directory of this repository);
2. Execute 
   ```console
   foo@bar:~$ DOCKER_BUILDKIT=1 docker build -t dishonest_mia .
   ```
   This downloads the environment and python packages.
3. The following command launches the container and the Jupyter notebook (since it is launced with no browser support, the information about the token is also displayed at this point):
   ```console
   foo@bar:~$  docker run --network host -e port=8888 dishonest_mia
   ```
4. Open a browser (in the local machine) and browse to [http://localhost:8888](http://localhost:8888). The login token can be found in the docker output.

## Commandline Usage
To reproduce all results in the paper, `main.py` can be used:
 1. Set up a Python environment that satisfies `requirements.txt`.
 2. Call
    ```console
    foo@bar:~$ python main.py 'optimizer=glob(*)' 'dataset=glob(*)' "param.batches=1,4,16,64,256"  # different batchsizes
    foo@bar:~$ python main.py 'optimizer=glob(*)' 'dataset=glob(*)' "param.epochs=1,2,4"           # different no. of epochs
    foo@bar:~$ python main.py 'optimizer=glob(*)' 'dataset=glob(*)' "param.top_j=1,2,8,16"         # different M
    ```
 3. Results are saved in `output/`.
 4. Call
    ```console
    foo@bar:~$ python process.py
    ```
 5. Results are available in `output/results.json`. They can be explored using the `main.ipynb` notebook.

_You might need to download the CelebA dataset to `datasets/` manually due to a [bug](https://github.com/pytorch/vision/issues/2262)._


## Files and directories

| File/Directory        | Description                                                                                                           |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------|
| `main.py`             | Main executable. Configuration can be found in `conf/`. By default, results are saved in `*.json` files in `output/`. |
| `process.py`          | script that merges the results in `output/` into `output/results.json`.                                               |
| `client.py`           | Implements the `flower`-client.                                                                                       |
| `dishonest_server.py` | Implements the dishonest server as a `flwr.flwr.server.strategy.Strategy`.                                            |
| `util.py`             | Utility functions.                                                                                                    |
| `requirements.txt`    | Python requirements.                                                                                                  |
| `results.json.orig`   | Original `results.json`.                                                                                              |
| `README.md`           | You are here.                                                                                                         |
| `workaround.py`       | Workaround for a [missing feature](https://github.com/adap/flower/pull/1115) in the current `flower` implementation.                                               |                     
| `conf/`               | Configuration YAML files.                                                                                             |
| `Dockerfile`          | Allows to build a docker container to run jupyter.                                                                    |
| `main.ipynb`          | Juptyer notebook to run individual experiments and display the results.                                               |
| `duplicates.ipynb`    | Jupyter notebook to explore the duplicate images in CIFAR100 and CelebA.                                              |
| `duplicates.json`     | List of duplicates in training and/or testing sets of CIFAR100 and CelebA.                                            |


