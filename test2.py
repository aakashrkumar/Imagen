import matplotlib.pyplot as plt
import random
import tqdm

TRIALS = 1000000
TOSSES = 1000
trials = []
for i in tqdm.tqdm(range(TRIALS)):
    trial = []
    for _ in range(TOSSES):
        trial.append(random.randint(0, 1))
    trials.append(trial)

for i, trial in enumerate(trials):
    print(f"Trial {i + 1}: {sum(trial) / TOSSES}")
# make a histogram of the results
plt.hist([sum(trial) / TOSSES for trial in trials])
plt.show()