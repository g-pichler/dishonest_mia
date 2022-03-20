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
The repository allows the use to enjoy a containarized version of the code taking care of the creation and setting up of a conda environment with all the required packages. In case docker is not installed, we suggest the following [guide](https://docs.docker.com/engine/install/). The following guide has been written considering a Linux based system. 
Steps required to run the Jupyter Notebook in a docker container:
1. Navigate to the folder where the Dockerfile is stored;
2. Execute 
   ```console
   foo@bar:~$ DOCKER_BUILDKIT=1 docker build -t dishonest_mia .
   ``` 
3. The following command activates a virtual environment environment, installs the required packages and launches the Jupyter notebook (since it is launced with no browser support, the information about the token is also displayed at this point):
   ```console
   foo@bar:~$  docker run --network host -e port=<port> dishonest_mia
   ```
4. (optional) In case the docker container is running remotely the next step is required in the local machine:
   ```console
   foo@bar:~$ ssh -N -f -L localhost:<local_port>:localhost:<port> user@remote_server
   ```
5. Open a browser (in the local machine) and browse to localhost:<local_port>.

#### Useful heads-up concerning docker
- If the following error occurs 
   ```
     Got permission denied while trying to connect to the Docker daemon socket ... permission denied
   ```
   the solution can be found [here](https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket);
- We have noticed that docker tends to create quite some dirt in the dedicated folder, some useful commands to clean up can be found [here](https://stackoverflow.com/questions/27853571/why-is-docker-image-eating-up-my-disk-space-that-is-not-used-by-docker), [here](https://docs.docker.com/engine/reference/commandline/rmi/), [here](https://docs.docker.com/engine/reference/commandline/system_prune/), and [here](https://gist.github.com/evanscottgray/8571828).

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


