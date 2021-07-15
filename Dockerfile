FROM bhugo/ddfacet:0.6.0
MAINTAINER Ben Hugo "bhugo@ska.ac.za"

#Copy killMS into the image
ADD killMS /src/killMS/killMS
ADD MANIFEST.in /src/killMS/MANIFEST.in
ADD setup.py /src/killMS/setup.py
ADD setup.cfg /src/killMS/setup.cfg
ADD README.md /src/killMS/README.md
ADD .git /src/killMS/.git
ADD .gitignore /src/killMS/.gitignore

WORKDIR /src
#build and install
RUN python3 -m pip install ./killMS
RUN kMS.py --help
RUN pybdsf --version

# set as entrypoint - user should be able to run docker run kmstag and get help printed
ENTRYPOINT ["kMS.py"]
CMD ["--help"]
