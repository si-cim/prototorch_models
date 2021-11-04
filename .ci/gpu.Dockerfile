FROM nvcr.io/nvidia/pytorch:21.10-py3

RUN adduser --uid 1000 jenkins

USER jenkins
