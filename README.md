# Defending Our Privacy With Backdoors

  <center>
  <img src="readme_images/idia_backdoor_defense.png" alt="False-Positive Examples"  height=300>
  </center>


Abstract: *The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. 
One of the concerns is that adversaries can extract information about the training data using privacy attacks. 
Unfortunately, the task of removing specific information from the models without sacrificing 
performance is not straightforward and has proven to be challenging.
We propose a rather easy yet effective defense based on backdoor attacks to remove private information such as names of individuals from models, and focus in this work on text encoders. 
Specifically, through strategic insertion of backdoors, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's name.
Our empirical results demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers.
Our approach provides not only a new "dual-use" perspective on backdoor attacks, but also presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.*    
[Arxiv Preprint](https://arxiv.org/abs/2310.08320)

# Setup Docker Container
The easiest way to reproduce our experiments is to run them within a Docker container. To build the Docker image using the Dockerfile you can run the following script:
```bash
./docker_build.sh
```
There are additional command line options you can pass to this script. To see all possible options run `./docker_build.sh -h`.

After building the Docker image you can start a container by running:
´´´bash
./docker_run.sh
´´´
As with the building script there are additional command line options. For further information run `./docker_run.sh -h`.

# Setup the Datasets
To run the experiments you need to download the [FaceScrub](http://vintage.winklerbros.net/facescrub.html) and place the folder into the ``