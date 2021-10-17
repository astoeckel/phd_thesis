FROM debian:10.10

# Install python and base fonts
RUN apt-get update \
	&& apt-get -y install \
		git \
		build-essential \
		python3-numpy python3-scipy python3-matplotlib python3-pip python3-h5py \
		ninja-build

RUN pip3 install \
	tqdm==4.59.0 \
	cython==0.29.22 \
	tensorflow==2.5.0-dev20210114

RUN mkdir -p /opt

ENV PYTHONPATH=/work/code/lib/
