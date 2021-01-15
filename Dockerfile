FROM python:3.7
FROM continuumio/miniconda:latest

#WORKDIR /SICS_DEC

# Create the environment:
COPY sics.yml .
RUN conda env create --file sics.yml
RUN conda activate sics-gpu

EXPOSE 8080

RUN chmod a+x ./run.sh

ENTRYPOINT ["./run.sh"]