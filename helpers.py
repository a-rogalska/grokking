import matplotlib.pyplot as plt
import seaborn

def plot_attention_heatmap(data, x, y, head_id, ax):
    seaborn.heatmap(data, xticklabels=x, yticklabels=y, square=True, vmin=0.0, vmax=1.0, cbar=False, annot=True, fmt=".2f", ax=ax)
    ax.set_title(f'MHA head id = {head_id}')
    ax.tick_params(labelrotation=0)


def visualize_attention_helper(attention_weights, source_seq, target_seq, layer_id, epoch):
    num_columns = 2
    num_rows = 2
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(10, 10))  # prepare the figure and axes

    for head_id, head_attention_weights in enumerate(attention_weights):
        row_index = int(head_id / num_columns)
        column_index = head_id % num_columns
        plot_attention_heatmap(head_attention_weights, source_seq, target_seq, head_id, axs[row_index, column_index])

    fig.suptitle(f'Encoder layer {layer_id + 1}')
    plt.savefig(f'./figures/attention1/attention_{layer_id + 1}_epoch_{epoch}.png', dpi=150)
    plt.close()


def visualize_attention(model, source_seq, target_seq, epoch=''):
    encoder = model.transformer_encoder

    # Visualize encoder attention weights
    for layer_id, encoder_layer in enumerate(encoder.layers):
        # attention_weights shape = (B, NH, S, S), extract 0th batch and loop over NH (number of heads) MHA heads
        # S stands for maximum source token-sequence length
        attn_weights = encoder_layer.attention_weights.cpu().numpy()[0]

        visualize_attention_helper(attn_weights, source_seq, target_seq, layer_id, epoch)

def make_plot(data, title, xlabel, ylabel, path):
    for (x, y, label) in data:
        plt.plot(x, y, label=label)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.savefig(path, dpi=150)
    plt.close()