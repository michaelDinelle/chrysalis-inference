# Chrysalis in Docker

The API alone, at `http://localhost:8080/v1`, on an RX 7900 XTX.

```sh
docker run -d --device /dev/kfd --device /dev/dri -p 127.0.0.1:8080:8080 -v chrysalis:/models ghcr.io/michaeldinelle/chrysalis
```
