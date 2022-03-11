import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("train_output/history_final.csv")
df.plot(
    x="iter",
    y=["loss_total", "loss_xy", "loss_wh", "loss_obj", "loss_cls"],
    figsize=(8, 4),
    logy=True,
)
plt.show()
