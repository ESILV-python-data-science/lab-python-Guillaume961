# Question 1 & 2

run command : 
```
save cmd > classify_images.py --images-list MNIST_all.csv --features-only
```

# Question 3

run the save command : 
```
save cmd > classify_images.py --images-list MNIST_all.csv --features-only --save-features ft_save
```
run the load command :
```
load cmd > classify_images.py --load-features ft_save.pickle --features-only
```

# Question 4 & 5

run command : 
```
cmd > classify_images.py --load-features ft_save.pickle --nearest-neighbors 1
```







