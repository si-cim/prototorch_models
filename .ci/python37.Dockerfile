FROM python:3.7

RUN adduser --uid 1000 jenkins

USER jenkins
