# Run each experiment for at most 5h

# WGAN with no PID
timeout 5h python3 train_pid.py configs/cifar_pid0.yaml

# WGAN with PID, Integral component 0.1
timeout 5h python3 train_pid.py configs/cifar_pid0.1yaml

# WGAN with PID, Integral component 0.1
timeout 5h python3 train_pid.py configs/cifar_pid1.yaml

# WGAN with PID, Integral component 0.1
timeout 5h python3 train_pid.py configs/cifar_pid5.yaml

# WGAN with PID, Integral component 10
timeout 5h python3 train_pid.py configs/cifar_pid10.yaml