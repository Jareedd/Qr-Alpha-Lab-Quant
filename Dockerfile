FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-dashboard.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt -r requirements-dashboard.txt

COPY . /app

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
