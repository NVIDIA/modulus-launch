To build release image:

Insert next tag:
```
docker build -t modulus-launch --target deploy -f Dockerfile .. 
```

To build CI image:

Insert next tag based on YY.MM.ID
(E.g. December 2022, 3rd image has tag 22.12.03)
```
docker build -t modulus-launch-ci:xx.xx.xx --target ci -f Dockerfile .. 
```
