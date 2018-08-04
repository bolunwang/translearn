# TransLearn

### ABOUT

This repository contains code implementation of the paper "[With Great
Training Comes Great Vulnerability: Practical Attacks against Transfer
Learning](http://people.cs.uchicago.edu/~ravenben/publications/pdf/translearn-usenixsec18.pdf)",
at *USENIX Security 2018*.


### DEPENDENCIES

Code is implemented using a mixure of Keras and TensorFlow.
Following packages are used to perform the attack and setup the attack evaluation:

- `keras==2.2.0`
- `numpy==1.14.0`
- `tensorflow-gpu==1.8.0`
- `h5py==2.8.0`

The code is tested using Python 2.7.


### HOWTO

We include a sample [script](https://github.com/bolunwang/translearn/blob/master/pubfig65-vggface-trans-mimic-target-penalty-dssim.py)
that demonstrate how to perform the attack on the Face Recognition example
and evaluate the attack performance.

```
python pubfig65-vggface-trans-mimic-target-penalty-dssim.py
```

There are several parameters that need to be modified before running the code,
which is included in the "[PARAMETER](https://github.com/bolunwang/translearn/blob/master/pubfig65-vggface-trans-mimic-target-penalty-dssim.py#L19-L54)"
section of the script.

1. Model files of the Teacher and Student need to be downloaded using the
following link, and placed at the correct path. Model files are specified by
`TEACHER_MODEL_FILE` and `STUDENT_MODEL_FILE`.
2. We included a sample data file, which includes 1 image for each label in
the Student model. Download the data file using the following [link](),
and place it under the same folder.
3. If you are using GPU, you need to specify which GPU you want to use for
the attack. This this specified by the `DEVICE` variable.


### DATASETS

Below is the list of datasets we used in the paper.

- **PubFig**: This dataset is used to train the Face Recognition model in the
paper. The detailed information about this dataset is included in
[this](http://vision.seas.harvard.edu/pubfig83/) page. We use a specific
[version](http://ic.unicamp.br/~chiachia/resources/pubfig83-aligned/)
of the dataset, where images are aligned.
- **CASIA Iris**: This dataset is used to train the Iris Recognition task.
Detailed information is included in [this](http://biometrics.idealtest.org/)
page.
- **GTSRB**: This dataset is used to train the Traffic Sign Recognition model.
Detailed information could be found
[here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
- **VGG Flower**: This dataset is used to train the Flower Recognition model.
Detailed information and download link could be found
[here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).
