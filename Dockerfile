ARG DEALII_IMAGE_VERSION="master"

FROM dealii/dealii:${DEALII_IMAGE_VERSION}-noble as builder
USER root

WORKDIR /app

COPY . .

RUN rm -rf build* && \ 
    cmake --preset=release && \
    cmake --build build-release -j$(nproc)

ENTRYPOINT ["/app/build-release/screened-poisson-fe"]
