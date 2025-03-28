## eXtreme Multi-Label Ranking for Entity Linking - XMR4EL

Following PECOS(https://github.com/amzn/pecos/tree/mainline) pipeline, this project tries to implement any kind of model into it. 

With the use of an Hierarchical Tree, we can try different options of vectorizers, transformers, clustering and classifier models to test our data.

Even thou this project has as main vision the use of it in Entity Linking, it can be used with any data, that uses as tool eXtreme Multi-Label Ranking (XMR) to work.

Paper: (Not made yet)

### Requirements and Installation

* Python (3.12)

All the packages used are displayed in the requirements.txt.

### CUDA Version

* CUDA (11.4 - 11.8)

Right now all the gpu models are from rapids.ai, and are tested in an docker enviroment, given in the dockerfile.


