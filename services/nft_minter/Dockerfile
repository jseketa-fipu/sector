FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/services/nft_minter

ENV PORT=9100
EXPOSE 9100

CMD ["python", "-m", "services.nft_minter.main"]
