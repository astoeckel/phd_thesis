FROM debian:10.10

# Install python and base fonts
RUN apt-get update \
	&& apt-get -y install \
		git \
		build-essential \
		python3-numpy python3-scipy python3-matplotlib python3-pip python3-h5py \
		libgsl-dev \
		ninja-build

RUN pip3 install meson==0.57.1 pytry==0.9.2 tqdm==4.59.0 nengo==3.1.0 nengo-extras==0.4.0 cython==0.29.22 brian2==2.4.2

RUN mkdir -p /opt \
    mkdir -p /.cython \
    chmod 777 /.cython

RUN git clone https://github.com/astoeckel/libbioneuronqp /opt/libbioneuronqp \
	&& cd /opt/libbioneuronqp \
	&& git checkout 7a788b59fb465915d32954639b512497b4c70259 \
	&& git submodule init \
	&& git submodule update \
	&& mkdir build \
	&& cd build \
	&& meson .. \
	&& ninja install \
	&& cd .. \
	&& pip3 install -e .

RUN git clone https://github.com/astoeckel/nengo-bio /opt/nengo-bio \
	&& cd /opt/nengo-bio \
	&& git checkout 04e992684aeb691b6d394e1a3a6c3cc5a536dc79 \
	&& pip3 install -e .

ENV LD_LIBRARY_PATH=/opt/libbioneuronqp/build/

