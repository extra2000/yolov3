# yolov3

YOLOv3 implemented in TensorFlow 1.x.


## Building Docs

```
chcon -R -v -t container_file_t ../yolov3
podman build -t extra2000/yolov3-tf-docs -f Dockerfile.sphinx .
podman run -it --rm -v ./docs:/opt/docs:rw localhost/extra2000/yolov3-tf-docs make clean html
podman run --rm -p 8080:80 -v ./docs/build/html:/usr/local/apache2/htdocs:ro docker.io/library/httpd:2.4
```


## Known Issues

* TensorFlow ROCm - Training failed with Nan.
* Raspberry Pi - Incomplete documentation because require a lot of work to build TensorFlow from source.
