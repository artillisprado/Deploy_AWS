###############
# BUILD IMAGE #
###############
FROM python:3.8
RUN python -m pip install --upgrade pip
WORKDIR /home/odg/sabia
COPY . .
RUN pip install -r ./file.txt
EXPOSE 8501
ENTRYPOINT [ "streamlit", "run" ]
CMD ["graficos-streamlit.py"]