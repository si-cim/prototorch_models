FROM python:3.8

RUN adduser --uid 1000 jenkins

USER jenkins
