FROM debian:10.10

# Install python and base fonts
RUN apt-get update \
	&& apt-get -y install \
		git \
		python3-numpy python3-scipy python3-pip python3-h5py

RUN pip3 install \
	tqdm==4.59.0 \
	nengo==3.1.0 \
	sympy==1.7.1

RUN mkdir -p /opt

RUN git clone https://github.com/astoeckel/pykinsim /opt/pykinsim \
	&& cd /opt/pykinsim \
	&& git checkout 5aaaa6ca5d05c7d8d15869f475a6c0633fc024a9 \
	&& pip3 install -e .


