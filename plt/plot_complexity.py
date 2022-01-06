import numpy as np
import matplotlib.pyplot as plt


fixed = 'm=5'  # 'j=30', 'm=5'
datas = [
    'RL-GNN_complexity_fixed_{}.npy'.format(fixed),
    'RL-GNN_complexity_fixed_{}_reimplement.npy'.format(fixed),
    'L2D_complexity_fixed_{}.npy'.format(fixed),
    'L2S_complexity_fixed_{}_[500].npy'.format(fixed)
]

times_for_plot = []

for data in datas:
    times_for_plot.append(np.load(data))

x_labels = [str(5+5*i) for i in range(times_for_plot[1].shape[0])]

# plot parameters
x_label_scale = 15
y_label_scale = 15
anchor_text_size = 15
title_size = 15
show = False
save = True
save_file_type = '.png'


# obj1 = times_for_plot[0]  # RL-GNN
obj2 = times_for_plot[1]  # RL-GNN Reimplement
# obj3 = times_for_plot[2]  # L2D
# obj4 = times_for_plot[3][:, 0].reshape(-1)  # ours-500

# plotting...
plt.figure(figsize=(5.3, 5))
# plt.xlabel('Number of jobs {}'.format(r'$n$'), {'size': x_label_scale})
plt.xlabel('Number of machines {}'.format(r'$m$'), {'size': x_label_scale})

plt.ylabel('Seconds', {'size': y_label_scale})
plt.grid()
x = np.array(x_labels)
# plt.plot(x, obj1, color='tab:green', marker="o", label='RL-GNN')
plt.plot(x, obj2, color='tab:red', marker="s", label='RL-GNN-Re')
# plt.plot(x, obj3, color='tab:brown', linestyle="--", marker="v", label='L2D')
# plt.plot(x, obj4, color='tab:blue', linestyle="--", marker="v", label='L2S-500')

plt.tight_layout()
plt.legend(fontsize=anchor_text_size)
if save:
    plt.savefig('./complexity_analysis_{}{}'.format(fixed, save_file_type))
if show:
    plt.show()