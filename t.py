import re
import matplotlib.pyplot as plt

# 正则表达式模式，用于匹配loss值
pattern = re.compile(r"\{'loss': ([\d\.]+), .*?\}")
# 读取日志文件并提取loss值
def read_log_file(file_path):
    steps = []
    losses = []
    t=1
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            match = pattern.search(line)
            if match:
                loss = float(match.group(1))
                steps.append(t)
                losses.append(loss)
                t+=1
    return steps, losses

# Paths to the log files
log_files = ['20481.log','20482.log','20483.log']

# Plotting the losses from each log file
plt.figure(figsize=(10, 6))
for log_file in log_files:
    steps, losses = read_log_file(log_file)
    plt.plot(steps, losses, label=log_file)

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss vs Step for Multiple Log Files')
plt.legend()
plt.grid(True)
plt.savefig('loss_vs_step.png')
