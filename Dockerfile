FROM bhugo/ddfacet:0.7.0
MAINTAINER Ben Hugo "bhugo@ska.ac.za"

#Copy killMS into the image
ADD killMS /opt/killMS/killMS
ADD pyproject.toml /opt/killMS/pyproject.toml
ADD README.md /opt/killMS/README.md
ADD .git /opt/killMS/.git
ADD .gitignore /opt/killMS/.gitignore

WORKDIR /opt
#build and install
RUN . /opt/venv/bin/activate && python3 -m pip install ./killMS

RUN DDF.py --help
RUN MakeMask.py --help
RUN MakeCatalog.py --help
RUN MakeModel.py --help
RUN MaskDicoModel.py --help
RUN ClusterCat.py --help
RUN kMS.py --help
RUN pybdsf --version

# set as entrypoint - user should be able to run docker run kmstag and get help printed
ENTRYPOINT ["kMS.py"]
CMD ["--help"]
